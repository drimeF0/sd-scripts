"""
Explorative Modeling (XM) for Anima training.

Implements the best-of-k noise exploration strategy from:
https://arxiv.org/abs/2607.27372

Adapted from: https://github.com/alexiglad/ebt (XM repository)
"""

import math
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

import library.loss as loss_util
from library import flux_train_utils, anima_train_utils


def add_xm_arguments(parser):
    """
    Add Explorative Modeling (XM) arguments to the argument parser.

    Usage::
        parser = argparse.ArgumentParser()
        add_xm_arguments(parser)
    """
    parser.add_argument(
        "--xm_best_of_k",
        type=int,
        default=1,
        help="Explorative Modeling best-of-k for computing loss. "
        "1 = standard training (no exploration). "
        ">1 enables exploration over multiple noise candidates, "
        "selecting the one with minimum loss for backpropagation.",
    )
    parser.add_argument(
        "--xm_chunk_bs_mult",
        type=int,
        default=4,
        help="Explorative Modeling chunk batch size multiplier when doing "
        "best-of-k exploration. Higher values = more candidates processed "
        "in parallel, uses more memory.",
    )
    parser.add_argument(
        "--xm_save_mem_mode",
        action="store_true",
        default=False,
        help="Save memory during XM exploration: first explore without "
        "gradients to find the best noise, then recompute with gradients "
        "for the best candidate only.",
    )
    parser.add_argument(
        "--xm_no_save_mem_mode",
        action="store_false",
        dest="xm_save_mem_mode",
        help="Disable save_mem_mode for XM (uses more memory but avoids "
        "the second forward pass).",
    )
    parser.add_argument(
        "--xm_debug_mode",
        action="store_true",
        default=False,
        help="Debug XM mode with assertions to verify that the recomputed "
        "loss matches the explored loss.",
    )
    return parser


def validate_xm_args(args) -> None:
    """Validate combinations after CLI and config-file values are merged."""
    if args.xm_best_of_k < 1:
        raise ValueError(f"--xm_best_of_k must be at least 1, got {args.xm_best_of_k}")
    if args.xm_chunk_bs_mult < 1:
        raise ValueError(f"--xm_chunk_bs_mult must be at least 1, got {args.xm_chunk_bs_mult}")
    if args.xm_best_of_k > 1 and args.xm_debug_mode and not args.xm_save_mem_mode:
        raise ValueError("--xm_debug_mode requires --xm_save_mem_mode")
    if args.xm_best_of_k > 1 and not args.xm_save_mem_mode and args.xm_chunk_bs_mult < args.xm_best_of_k:
        raise ValueError(
            "without --xm_save_mem_mode, --xm_chunk_bs_mult must be at least --xm_best_of_k "
            "so the differentiable candidates fit in one chunk"
        )


def _expand_batch(value: Any, multiplier: int) -> Any:
    """Repeat tensors in candidate-major order while preserving containers/None."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return torch.cat([value] * multiplier, dim=0)
    if isinstance(value, tuple):
        return tuple(_expand_batch(item, multiplier) for item in value)
    if isinstance(value, list):
        return [_expand_batch(item, multiplier) for item in value]
    if isinstance(value, dict):
        return {key: _expand_batch(item, multiplier) for key, item in value.items()}
    raise TypeError(f"XM conditions must be tensors or containers of tensors, got {type(value)!r}")


def _index_batch(value: Any, indices: torch.Tensor) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[indices]
    if isinstance(value, tuple):
        return tuple(_index_batch(item, indices) for item in value)
    if isinstance(value, list):
        return [_index_batch(item, indices) for item in value]
    if isinstance(value, dict):
        return {key: _index_batch(item, indices) for key, item in value.items()}
    raise TypeError(f"XM predictions must be tensors or containers of tensors, got {type(value)!r}")


def _masked_update_batch(destination: Any, source: Any, mask: torch.Tensor) -> None:
    if destination is None:
        return
    if isinstance(destination, torch.Tensor):
        destination[mask] = source[mask]
        return
    if isinstance(destination, (tuple, list)):
        for destination_item, source_item in zip(destination, source):
            _masked_update_batch(destination_item, source_item, mask)
        return
    if isinstance(destination, dict):
        for key in destination:
            _masked_update_batch(destination[key], source[key], mask)
        return
    raise TypeError(f"XM predictions must be tensors or containers of tensors, got {type(destination)!r}")


def _allclose_nested(first: Any, second: Any, *, rtol: float, atol: float) -> bool:
    if first is None or second is None:
        return first is second
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.allclose(first, second, rtol=rtol, atol=atol)
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(
            _allclose_nested(a, b, rtol=rtol, atol=atol) for a, b in zip(first, second)
        )
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _allclose_nested(first[key], second[key], rtol=rtol, atol=atol) for key in first
        )
    return first == second


def _prepare_loss_mask(args, batch, reference: torch.Tensor) -> Optional[torch.Tensor]:
    """Build the same spatial mask used by the regular training path."""
    if batch is None:
        return None

    alpha_masks = batch.get("alpha_masks")
    use_mask = bool(getattr(args, "masked_loss", False)) or alpha_masks is not None
    if not use_mask:
        return None

    if "conditioning_images" in batch:
        mask = batch["conditioning_images"][:, :1] / 2 + 0.5
    elif alpha_masks is not None:
        mask = alpha_masks.unsqueeze(1)
    else:
        return None

    mask = mask.to(device=reference.device, dtype=torch.float32)
    return F.interpolate(mask, size=reference.shape[-2:], mode="area")


def xm_chunked_best_of_k(
    loss_calc_wrapper: Callable,
    conditions,
    gt_samples: torch.Tensor,
    best_of_k: int,
    max_chunk_bs_mult: int,
    save_mem_mode: bool = True,
    debug_save_mem_mode: bool = False,
    not_training: bool = False,
    **loss_calc_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Parallelizes Explorative Modeling over chunks to find the best-of-k loss mode.

    This function explores multiple potential solutions in parallel (chunks) and
    selects the one with the minimum loss. Chunks allow simulating multiple batches
    at once for parallelization.

    Adapted from XM (https://github.com/alexiglad/ebt).

    Args:
        loss_calc_wrapper: Callable that takes
            ``(conditions, gt_samples, learning, rand_inputs, rand_seeds, **kwargs)``
            and returns ``(losses, predictions)``.
        conditions: Conditions for the model (tuple of tensors or single tensor).
        gt_samples: Ground truth samples (latents).
        best_of_k: Number of candidates to explore to find the lowest loss mode.
        max_chunk_bs_mult: Max batch size multiplier for each chunk.
        save_mem_mode: If True, minimizes gradients during the exploration loop to
            save memory, then recomputes the loss with full gradients for the best
            candidate. Defaults to True.
        debug_save_mem_mode: If True, verifies that recomputed tensors in
            save_mem_mode are approximately the same as the originals.
        not_training: If True, means that the grad will never be tracked as the
            model is not being trained. Useful for val/testing.

    Returns:
        Tuple of (best_losses, best_predictions):
            - best_losses: The minimum losses found for each batch element.
            - best_predictions: The predictions corresponding to the minimum losses
              (or None if not available).
    """
    # Short-circuit for best_of_k=1: skip chunking overhead
    if best_of_k == 1:
        learning_direct = not not_training
        rand_inputs = torch.randn_like(gt_samples)
        rand_seeds = torch.randint(0, 2147483647, (gt_samples.shape[0],), device=gt_samples.device)
        losses, predictions = loss_calc_wrapper(
            conditions,
            gt_samples,
            learning=learning_direct,
            rand_inputs=rand_inputs,
            rand_seeds=rand_seeds,
            **loss_calc_kwargs,
        )
        return losses, predictions

    # Prepare chunking variables
    if debug_save_mem_mode:
        assert save_mem_mode, "debug_mode can only be used when debug_save_mem_mode is set"
    if not save_mem_mode and not not_training:
        assert (
            max_chunk_bs_mult >= best_of_k
        ), "save_mem_mode=False requires max_chunk_bs_mult >= best_of_k so all candidates fit in a single chunk"

    regular_bs = gt_samples.shape[0]
    assert max_chunk_bs_mult >= 1, "need to be exploring with a max_chunk_bs_mult >= 1"
    assert best_of_k >= 1, "best_of_k needs to be >= 1 for this to work"

    first_iter = True
    total_exploration_bs = regular_bs * best_of_k
    max_chunk_bs = max_chunk_bs_mult * regular_bs
    for_loop_iters = math.ceil(total_exploration_bs / max_chunk_bs)
    remaining_bs = total_exploration_bs
    learning = not save_mem_mode if not not_training else False
    best_predictions = None

    with torch.set_grad_enabled(learning):
        for _ in range(for_loop_iters):
            # Prepare per iteration chunking
            curr_chunk_bs = remaining_bs if remaining_bs <= max_chunk_bs else max_chunk_bs
            assert curr_chunk_bs % regular_bs == 0, "need to use a chunk divisible by regular bs"
            remaining_bs = remaining_bs - curr_chunk_bs
            this_chunk_bs_mult = curr_chunk_bs // regular_bs

            # Prepare random inputs
            rand_inputs = torch.randn(
                (curr_chunk_bs, *gt_samples.shape[1:]), device=gt_samples.device, dtype=gt_samples.dtype
            )
            rand_seeds = torch.randint(0, 2147483647, (curr_chunk_bs,), device=gt_samples.device)

            # Prepare conditions and ground truth
            conditions_expanded = _expand_batch(conditions, this_chunk_bs_mult)
            gt_samples_expanded = _expand_batch(gt_samples, this_chunk_bs_mult)

            # Compute losses
            losses, predictions = loss_calc_wrapper(
                conditions_expanded,
                gt_samples_expanded,
                learning=learning,
                rand_inputs=rand_inputs,
                rand_seeds=rand_seeds,
                **loss_calc_kwargs,
            )

            # Best-of-k along chunk
            chunk_losses_reshaped = losses.reshape(this_chunk_bs_mult, regular_bs)
            chunk_min_losses, chunk_min_indices = chunk_losses_reshaped.min(dim=0)

            # Convert to flat indices
            flat_indices = chunk_min_indices * regular_bs + torch.arange(regular_bs, device=gt_samples.device)

            chunk_best_rand_inputs = rand_inputs[flat_indices]
            chunk_best_rand_seeds = rand_seeds[flat_indices]

            save_best_predictions = (
                (debug_save_mem_mode and predictions is not None)
                if save_mem_mode
                else (predictions is not None)
            )

            if save_best_predictions:
                chunk_best_predictions = _index_batch(predictions, flat_indices)

            if first_iter:
                first_iter = False
                best_rand_inputs = chunk_best_rand_inputs
                best_rand_seeds = chunk_best_rand_seeds
                best_losses = chunk_min_losses
                if save_best_predictions:
                    best_predictions = chunk_best_predictions
            else:
                replacement_mask = chunk_min_losses < best_losses
                if replacement_mask.any():
                    best_rand_inputs[replacement_mask] = chunk_best_rand_inputs[replacement_mask]
                    best_rand_seeds[replacement_mask] = chunk_best_rand_seeds[replacement_mask]
                    best_losses[replacement_mask] = chunk_min_losses[replacement_mask]
                    if save_best_predictions:
                        _masked_update_batch(best_predictions, chunk_best_predictions, replacement_mask)

    # If save_mem_mode, do final forward with gradients
    if save_mem_mode:
        learning = True if not not_training else False
        torch.clear_autocast_cache()
        final_losses, final_predictions = loss_calc_wrapper(
            conditions,
            gt_samples,
            learning=learning,
            rand_inputs=best_rand_inputs,
            rand_seeds=best_rand_seeds,
            **loss_calc_kwargs,
        )

        if debug_save_mem_mode:
            if best_predictions is not None:
                assert _allclose_nested(best_predictions, final_predictions, rtol=1e-5, atol=1e-8), (
                    "predictions did not reproduce when doing 2nd round for comp graph"
                )
            assert torch.allclose(best_losses, final_losses, rtol=1e-5, atol=1e-8), (
                "losses did not reproduce when doing 2nd round for comp graph"
            )

        return final_losses, final_predictions
    else:
        # Already computed best_losses and best_predictions
        return best_losses, best_predictions


def compute_anima_xm_loss(
    args,
    dit,
    latents,
    noise_scheduler,
    prompt_embeds,
    attn_mask,
    t5_input_ids,
    t5_attn_mask,
    padding_mask,
    accelerator,
    weight_dtype,
    batch=None,
):
    """
    Compute Anima loss with Explorative Modeling (best-of-k noise exploration).

    For ``xm_best_of_k=1``, this is equivalent to standard training.
    For ``xm_best_of_k>1``, explores K noise samples per batch element and picks
    the one with minimum loss for backpropagation.

    Args:
        args: Training arguments (must include xm_* attributes).
        dit: The Anima DiT model.
        latents: Latent representations (B, C, H, W).
        noise_scheduler: Noise scheduler for timestep generation.
        prompt_embeds: Text prompt embeddings.
        attn_mask: Attention mask for text embeddings.
        t5_input_ids: T5 input token IDs.
        t5_attn_mask: T5 attention mask.
        padding_mask: Padding mask for latents.
        accelerator: HuggingFace Accelerator.
        weight_dtype: Weight dtype for the model.
        batch: Optional batch dict (used for loss_weights, alpha_masks, etc.).

    Returns:
        Scalar loss tensor.
    """
    best_of_k = args.xm_best_of_k
    chunk_bs_mult = args.xm_chunk_bs_mult
    save_mem_mode = args.xm_save_mem_mode
    debug_save_mem_mode = args.xm_debug_mode

    # Short-circuit: standard training (no exploration)
    if best_of_k <= 1:
        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        timesteps = timesteps / 1000.0

        # NaN checks
        if torch.any(torch.isnan(noisy_model_input)):
            accelerator.print("NaN found in noisy_model_input, replacing with zeros")
            noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

        # Forward pass
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D
        with accelerator.autocast():
            model_pred = dit(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                source_attention_mask=attn_mask,
                t5_input_ids=t5_input_ids,
                t5_attn_mask=t5_attn_mask,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D

        # Compute loss
        target = noise - latents
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

        huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, None)
        loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
        if batch is not None and (args.masked_loss or batch.get("alpha_masks") is not None):
            from library.custom_train_functions import apply_masked_loss
            loss = apply_masked_loss(loss, batch)

        if weighting is not None:
            loss = loss * weighting

        loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)

        loss_weights = (
            batch.get("loss_weights", torch.ones(loss.shape[0], device=loss.device))
            if batch
            else torch.ones(loss.shape[0], device=loss.device)
        )
        loss = loss * loss_weights.to(loss.device)
        return loss.mean()

    # === XM: explore over K noise candidates ===
    # Step 1: Get timesteps and sigmas once (shared across all candidates)
    # Use a dummy call to extract timesteps and sigmas
    dummy_noise = torch.zeros_like(latents)
    with torch.no_grad():
        _, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, dummy_noise, accelerator.device, weight_dtype
        )
    timesteps = timesteps / 1000.0

    # Compute weighting once
    weighting = anima_train_utils.compute_loss_weighting_for_anima(
        weighting_scheme=args.weighting_scheme, sigmas=sigmas
    )
    # Get loss_weights from batch
    loss_weights = (
        batch.get("loss_weights", torch.ones(latents.shape[0], device=latents.device))
        if batch
        else torch.ones(latents.shape[0], device=latents.device)
    ).to(latents.device)
    loss_mask = _prepare_loss_mask(args, batch, latents)

    # Handle IP noise gamma
    ip_noise_xi = None
    ip_noise_gamma_val = None
    if args.ip_noise_gamma:
        ip_noise_xi = torch.randn_like(latents)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma_val = torch.rand(1, device=latents.device, dtype=weight_dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma_val = args.ip_noise_gamma

    # Build conditions tuple (everything except latents and noise)
    # These will be expanded by xm_chunked_best_of_k for each chunk
    conditions = (
        timesteps,
        sigmas,
        prompt_embeds,
        attn_mask,
        t5_input_ids,
        t5_attn_mask,
        padding_mask,
        weighting,
        loss_weights,
        loss_mask,
        ip_noise_xi,
    )

    def loss_calc_wrapper(cond_expanded, gt_expanded, learning, rand_inputs, rand_seeds, **kwargs):
        """Compute loss for a batch of noise candidates (expanded by chunk_mult)."""
        t, sig, pe, am, t5_ids, t5_am, pm, w, lw, mask, ip_xi = cond_expanded

        # Compute noisy model input
        if ip_xi is not None:
            noisy = (1.0 - sig) * gt_expanded + sig * (rand_inputs + ip_noise_gamma_val * ip_xi)
        else:
            noisy = (1.0 - sig) * gt_expanded + sig * rand_inputs

        # Forward pass
        noisy_5d = noisy.unsqueeze(2)  # 4D to 5D
        with torch.set_grad_enabled(learning), accelerator.autocast():
            model_pred = dit(
                noisy_5d,
                t,
                pe,
                padding_mask=pm,
                source_attention_mask=am,
                t5_input_ids=t5_ids,
                t5_attn_mask=t5_am,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D

        # Target: noise - latents
        target = rand_inputs - gt_expanded

        # Loss
        huber_c = loss_util.get_huber_threshold_if_needed(args, t, noise_scheduler)
        loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)

        if w is not None:
            loss = loss * w
        if mask is not None:
            loss = loss * mask

        loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)

        loss = loss * lw

        return loss, model_pred

    # Run chunked best-of-k exploration
    best_losses, _ = xm_chunked_best_of_k(
        loss_calc_wrapper,
        conditions,
        latents,
        best_of_k,
        chunk_bs_mult,
        save_mem_mode=save_mem_mode,
        debug_save_mem_mode=debug_save_mem_mode,
        not_training=False,
    )

    return best_losses.mean()


def compute_anima_xm_loss_for_network(
    args,
    accelerator,
    noise_scheduler,
    latents,
    batch,
    text_encoder_conds,
    unet,
    weight_dtype,
    train_unet,
    is_train=True,
):
    """
    Compute XM loss for Anima network (LoRA) training.

    This is used inside ``get_noise_pred_and_target`` for network training.
    When ``xm_best_of_k <= 1``, falls back to standard single-noise computation.

    Returns:
        Tuple of (model_pred, target, timesteps, weighting) for the best candidate.
    """
    best_of_k = args.xm_best_of_k
    chunk_bs_mult = args.xm_chunk_bs_mult
    save_mem_mode = args.xm_save_mem_mode
    debug_save_mem_mode = args.xm_debug_mode

    bs = latents.shape[0]
    device = latents.device
    # Unpack text encoder conditions
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[:4]

    # Move to device
    prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
    attn_mask = attn_mask.to(device) if attn_mask is not None else None
    t5_input_ids = t5_input_ids.to(device, dtype=torch.long) if t5_input_ids is not None else None
    t5_attn_mask = t5_attn_mask.to(device) if t5_attn_mask is not None else None

    # Create padding mask
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]
    padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=device)

    # Short-circuit: standard training
    if best_of_k <= 1:
        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, weight_dtype
        )
        timesteps = timesteps / 1000.0

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Forward
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D

        target = noise - latents
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=sigmas
        )

        return model_pred, target, timesteps, weighting

    # === XM: explore over K noise candidates ===
    # Get timesteps and sigmas once
    dummy_noise = torch.zeros_like(latents)
    with torch.no_grad():
        _, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, dummy_noise, device, weight_dtype
        )
    timesteps = timesteps / 1000.0

    # Weighting
    weighting = anima_train_utils.compute_loss_weighting_for_anima(
        weighting_scheme=args.weighting_scheme, sigmas=sigmas
    )
    loss_mask = _prepare_loss_mask(args, batch, latents)

    # IP noise is sampled once per original item and shared by all of its
    # candidates, matching the regular Anima noisy-input construction.
    ip_noise_xi = None
    ip_noise_gamma_val = None
    if args.ip_noise_gamma:
        ip_noise_xi = torch.randn_like(latents)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma_val = torch.rand(1, device=device, dtype=weight_dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma_val = args.ip_noise_gamma

    # Build conditions tuple
    conditions = (
        timesteps,
        sigmas,
        prompt_embeds,
        attn_mask,
        t5_input_ids,
        t5_attn_mask,
        padding_mask,
        weighting,
        loss_mask,
        ip_noise_xi,
    )

    def loss_calc_wrapper(cond_expanded, gt_expanded, learning, rand_inputs, rand_seeds, **kwargs):
        """Compute loss for expanded noise candidates. Returns (loss, (model_pred, target))."""
        t, sig, pe, am, t5_ids, t5_am, pm, w, mask, ip_xi = cond_expanded

        # Compute noisy model input
        effective_noise = rand_inputs if ip_xi is None else rand_inputs + ip_noise_gamma_val * ip_xi
        noisy = (1.0 - sig) * gt_expanded + sig * effective_noise

        # Reentrant checkpointing needs at least one grad-requiring input when
        # only the attached network (for example LoRA) is trainable.
        if learning and args.gradient_checkpointing:
            noisy.requires_grad_(True)
            for condition in (pe, am, t5_am):
                if condition is not None and condition.dtype.is_floating_point:
                    condition.requires_grad_(True)

        # Forward
        noisy_5d = noisy.unsqueeze(2)  # 4D to 5D
        with torch.set_grad_enabled(learning), accelerator.autocast():
            model_pred = unet(
                noisy_5d,
                t,
                pe,
                padding_mask=pm,
                target_input_ids=t5_ids,
                target_attention_mask=t5_am,
                source_attention_mask=am,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D

        # Target: noise - latents
        target = rand_inputs - gt_expanded

        # Loss
        huber_c = loss_util.get_huber_threshold_if_needed(args, t, noise_scheduler)
        loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
        if w is not None:
            loss = loss * w
        if mask is not None:
            loss = loss * mask
        loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)

        # The final save-memory recomputation must retain model_pred's graph;
        # the caller computes the actual training loss from this prediction.
        return loss, (model_pred, target)

    # Run chunked best-of-k exploration
    # The predictions are tuples (model_pred, target) for the best candidates
    _, best_predictions_tuple = xm_chunked_best_of_k(
        loss_calc_wrapper,
        conditions,
        latents,
        best_of_k,
        chunk_bs_mult,
        save_mem_mode=save_mem_mode,
        debug_save_mem_mode=debug_save_mem_mode,
        not_training=not is_train,
    )

    # Unpack the best predictions
    if best_predictions_tuple is not None:
        best_model_pred, best_target = best_predictions_tuple
    else:
        # Fallback: recompute with the same sigmas (shouldn't happen in practice)
        best_model_pred = torch.zeros_like(latents)
        best_target = torch.zeros_like(latents)

    return best_model_pred, best_target, timesteps, weighting
