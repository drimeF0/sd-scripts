"""
Explorative Modeling (XM) for Anima training.

Implements the best-of-k noise exploration strategy from:
https://arxiv.org/abs/2607.27372

Adapted from: https://github.com/alexiglad/ebt (XM repository)
"""

import math
import torch
from typing import Callable, Optional, Tuple

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
        default=True,
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
            rand_inputs = torch.randn((curr_chunk_bs, *gt_samples.shape[1:]), device=gt_samples.device)
            rand_seeds = torch.randint(0, 2147483647, (curr_chunk_bs,), device=gt_samples.device)

            # Prepare conditions and ground truth
            if isinstance(conditions, tuple):
                conditions_expanded = tuple(
                    torch.cat([c] * this_chunk_bs_mult, dim=0) for c in conditions
                )
            else:
                conditions_expanded = torch.cat([conditions] * this_chunk_bs_mult, dim=0)
            gt_samples_expanded = torch.cat([gt_samples] * this_chunk_bs_mult, dim=0)

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
                chunk_best_predictions = predictions[flat_indices]

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
                        best_predictions[replacement_mask] = chunk_best_predictions[replacement_mask]

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
                assert torch.allclose(best_predictions, final_predictions, rtol=1e-5, atol=1e-8), (
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
        if args.masked_loss and batch is not None and "alpha_masks" in batch and batch["alpha_masks"] is not None:
            from library.custom_train_functions import apply_masked_loss
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)

        if weighting is not None:
            loss = loss * weighting

        loss_weights = batch.get("loss_weights", torch.ones(loss.shape[0], device=loss.device)) if batch else torch.ones(loss.shape[0], device=loss.device)
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
    huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

    # Get loss_weights from batch
    loss_weights = batch.get("loss_weights", torch.ones(latents.shape[0], device=latents.device)) if batch else torch.ones(latents.shape[0], device=latents.device)

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
    )

    def loss_calc_wrapper(cond_expanded, gt_expanded, learning, rand_inputs, rand_seeds, **kwargs):
        """Compute loss for a batch of noise candidates (expanded by chunk_mult)."""
        t, sig, pe, am, t5_ids, t5_am, pm, w, lw = cond_expanded

        # Compute noisy model input
        if ip_noise_xi is not None:
            chunk_mult = rand_inputs.shape[0] // gt_expanded.shape[0]
            noisy = (1.0 - sig) * gt_expanded + sig * (rand_inputs + ip_noise_gamma_val * ip_noise_xi.repeat(chunk_mult, 1, 1, 1))
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
        loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
        loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)

        if w is not None:
            loss = loss * w

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
    dtype = latents.dtype

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
    huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)

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
    )

    def loss_calc_wrapper(cond_expanded, gt_expanded, learning, rand_inputs, rand_seeds, **kwargs):
        """Compute loss for expanded noise candidates. Returns (loss, (model_pred, target))."""
        t, sig, pe, am, t5_ids, t5_am, pm, w = cond_expanded

        # Compute noisy model input
        noisy = (1.0 - sig) * gt_expanded + sig * rand_inputs

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
        loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
        loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)
        if w is not None:
            loss = loss * w

        # Return (loss, (model_pred, target)) so xm_chunked_best_of_k can save both
        return loss, (model_pred.detach(), target.detach())

    # Run chunked best-of-k exploration
    # The predictions are tuples (model_pred, target) for the best candidates
    # Use a modified version of the exploration that handles tuple predictions
    best_losses, best_predictions_tuple = _xm_chunked_best_of_k_with_tuple(
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


def _xm_chunked_best_of_k_with_tuple(
    loss_calc_wrapper,
    conditions,
    gt_samples,
    best_of_k,
    max_chunk_bs_mult,
    save_mem_mode=True,
    debug_save_mem_mode=False,
    not_training=False,
    **loss_calc_kwargs,
):
    """
    Like xm_chunked_best_of_k but handles the case where predictions is a tuple (model_pred, target).
    This is used by compute_anima_xm_loss_for_network.
    """
    # Short-circuit for best_of_k=1
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
        assert save_mem_mode
    if not save_mem_mode and not not_training:
        assert max_chunk_bs_mult >= best_of_k

    regular_bs = gt_samples.shape[0]
    assert max_chunk_bs_mult >= 1
    assert best_of_k >= 1

    first_iter = True
    total_exploration_bs = regular_bs * best_of_k
    max_chunk_bs = max_chunk_bs_mult * regular_bs
    for_loop_iters = math.ceil(total_exploration_bs / max_chunk_bs)
    remaining_bs = total_exploration_bs
    learning = not save_mem_mode if not not_training else False
    best_predictions = None

    with torch.set_grad_enabled(learning):
        for _ in range(for_loop_iters):
            curr_chunk_bs = remaining_bs if remaining_bs <= max_chunk_bs else max_chunk_bs
            assert curr_chunk_bs % regular_bs == 0
            remaining_bs = remaining_bs - curr_chunk_bs
            this_chunk_bs_mult = curr_chunk_bs // regular_bs

            rand_inputs = torch.randn((curr_chunk_bs, *gt_samples.shape[1:]), device=gt_samples.device)
            rand_seeds = torch.randint(0, 2147483647, (curr_chunk_bs,), device=gt_samples.device)

            if isinstance(conditions, tuple):
                conditions_expanded = tuple(
                    torch.cat([c] * this_chunk_bs_mult, dim=0) for c in conditions
                )
            else:
                conditions_expanded = torch.cat([conditions] * this_chunk_bs_mult, dim=0)
            gt_samples_expanded = torch.cat([gt_samples] * this_chunk_bs_mult, dim=0)

            losses, preds_tuple = loss_calc_wrapper(
                conditions_expanded,
                gt_samples_expanded,
                learning=learning,
                rand_inputs=rand_inputs,
                rand_seeds=rand_seeds,
                **loss_calc_kwargs,
            )

            # preds_tuple is (model_pred, target) for all candidates
            chunk_losses_reshaped = losses.reshape(this_chunk_bs_mult, regular_bs)
            chunk_min_losses, chunk_min_indices = chunk_losses_reshaped.min(dim=0)

            flat_indices = chunk_min_indices * regular_bs + torch.arange(regular_bs, device=gt_samples.device)

            # Extract best model_pred and target for this chunk
            if preds_tuple is not None:
                chunk_model_pred, chunk_target = preds_tuple
                chunk_best_model_pred = chunk_model_pred[flat_indices]
                chunk_best_target = chunk_target[flat_indices]
            else:
                chunk_best_model_pred = None
                chunk_best_target = None

            if first_iter:
                first_iter = False
                best_rand_inputs = rand_inputs[flat_indices]
                best_rand_seeds = rand_seeds[flat_indices]
                best_losses = chunk_min_losses
                if chunk_best_model_pred is not None:
                    best_predictions = (chunk_best_model_pred, chunk_best_target)
            else:
                replacement_mask = chunk_min_losses < best_losses
                if replacement_mask.any():
                    best_rand_inputs[replacement_mask] = rand_inputs[flat_indices][replacement_mask]
                    best_rand_seeds[replacement_mask] = rand_seeds[flat_indices][replacement_mask]
                    best_losses[replacement_mask] = chunk_min_losses[replacement_mask]
                    if chunk_best_model_pred is not None:
                        best_model_pred_old, best_target_old = best_predictions
                        best_model_pred_old[replacement_mask] = chunk_best_model_pred[replacement_mask]
                        best_target_old[replacement_mask] = chunk_best_target[replacement_mask]
                        best_predictions = (best_model_pred_old, best_target_old)

    # If save_mem_mode, recompute with gradients
    if save_mem_mode:
        learning = True if not not_training else False
        torch.clear_autocast_cache()
        final_losses, final_preds_tuple = loss_calc_wrapper(
            conditions,
            gt_samples,
            learning=learning,
            rand_inputs=best_rand_inputs,
            rand_seeds=best_rand_seeds,
            **loss_calc_kwargs,
        )

        return final_losses, final_preds_tuple

    return best_losses, best_predictions