# Anima full finetune with DINOv3 image conditioning.
#
# Same as anima_train.py, but every DiT block is conditioned on DINOv3 image embeddings in addition to the Qwen3
# text encoder. A frozen DINOv3 ViT (default: the 0.3B ViT-L/16 mirror camenduru/dinov3-vitl16-pretrain-lvd1689m)
# embeds each training image; a small trainable projector maps those tokens into the cross-attention context space,
# where they are attended to by a decoupled cross-attention behind a zero-initialized per-block gate. Training
# therefore starts exactly at the pretrained model and opens the image branch as it learns.
#
# Everything expensive is cached ahead of training: VAE latents, Qwen3 text encoder outputs and DINOv3 embeddings.
#
# Example:
#   accelerate launch train_anima_dinov3.py \
#     --pretrained_model_name_or_path anima.safetensors --qwen3 qwen3_06b --vae qwen_image_vae.safetensors \
#     --dataset_config dataset.toml --output_dir out --output_name anima_dinov3 \
#     --cache_latents --cache_latents_to_disk \
#     --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
#     --dinov3_model_name_or_path camenduru/dinov3-vitl16-pretrain-lvd1689m \
#     --dinov3_image_size 224 --dinov3_token_mode patch --dinov3_pool_size 2 --dinov3_dropout_rate 0.1 \
#     --learning_rate 1e-5 --dinov3_projector_lr 1e-4 --optimizer_type adamw8bit \
#     --mixed_precision bf16 --gradient_checkpointing --max_train_epochs 10
#
# The DiT checkpoint stays in the stock Anima format; the DINOv3 projector and gates are saved next to it as
# <output_name>_dinov3_proj.safetensors.

import argparse
import copy
import gc
import math
import os
from multiprocessing import Value

import toml
import torch
from tqdm import tqdm

from library import flux_train_utils
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from accelerate.utils import set_seed
from library import (
    anima_dinov3,
    anima_train_utils,
    anima_utils,
    deepspeed_utils,
    sai_model_spec,
    strategy_anima,
    strategy_base,
)

import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.checkpoint_io as checkpoint_io
import library.config_util as config_util
import library.dataset as dataset_util
import library.logging_util as logging_util
import library.loss as loss_util
import library.optimizer as optimizer_util
import library.sampling as sampling

from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def save_dinov3_projector(args, model, encoder, epoch=None, global_step=None):
    """Save the projector next to the DiT checkpoints. The DiT itself is saved by anima_train_utils."""
    base_name = args.output_name if args.output_name is not None else "anima"
    if epoch is not None:
        file_name = f"{base_name}_dinov3_proj-{epoch:06d}.safetensors"
    elif global_step is not None:
        file_name = f"{base_name}_dinov3_proj-step{global_step:08d}.safetensors"
    else:
        file_name = f"{base_name}_dinov3_proj.safetensors"
    anima_dinov3.save_dinov3_projector(
        os.path.join(args.output_dir, file_name), model.dinov3_projector, encoder.cache_config
    )


def train(args):
    args_util.verify_training_args(args)
    accelerator_setup.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    flux_train_utils.log_timestep_sampling_info(args)

    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
        args.cache_text_encoder_outputs = True

    # DINOv3 embeddings reach the training loop through the text encoder output cache, so that cache is required.
    # They are stored in their own npz files, but must live on the same medium: the dataset reads either the npz
    # files or the in-memory ImageInfo, never a mix.
    assert (
        args.cache_text_encoder_outputs
    ), "--cache_text_encoder_outputs is required by train_anima_dinov3.py (DINOv3 embeddings ride along with it)"

    # The DiT is wrapped together with the projector, which breaks the DeepSpeed unwrap path used when saving.
    assert not args.deepspeed, "--deepspeed is not supported by train_anima_dinov3.py"

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning("cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
        args.gradient_checkpointing = True

    if args.unsloth_offload_checkpointing:
        if not args.gradient_checkpointing:
            logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
            args.gradient_checkpointing = True
        assert not args.cpu_offload_checkpointing, "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not args.unsloth_offload_checkpointing, "blocks_to_swap is not supported with unsloth_offload_checkpointing"

    if args.dit_offload_blocks is not None and args.dit_offload_blocks > 0:
        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ), "--dit_offload_blocks cannot be combined with --blocks_to_swap"
        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, (
            "--dit_offload_blocks requires at least 2 CUDA devices"
        )

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)

    # prepare caching strategy: must be set before preparing dataset
    if args.cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # prepare dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning("ignore following options because config file is found: {0}".format(", ".join(ignored)))
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = dataset_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)  # Qwen-Image VAE spatial downscale = 8 * patch size = 2

    if args.debug_dataset:
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
            anima_dinov3.AnimaDinoV3TextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, False, False
            )
        )
        train_dataset_group.set_current_strategies()
        dataset_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, either color_aug or random_crop cannot be used"

    assert train_dataset_group.is_text_encoder_output_cacheable(
        cache_supports_dropout=True
    ), "when caching text encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

    # prepare accelerator
    logger.info("prepare accelerator")
    accelerator = accelerator_setup.prepare_accelerator(args)

    # mixed precision dtype
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)

    # Cache DINOv3 embeddings. Runs first: in memory mode the text encoder caching pass picks them up from the
    # ImageInfo and stores both conditions in one tuple.
    # Not weight_dtype: DINOv3's activation outliers overflow float16 and yield NaN embeddings (see --dinov3_dtype).
    dinov3_encoder = anima_dinov3.DinoV3Encoder(
        model_name_or_path=args.dinov3_model_name_or_path,
        image_size=args.dinov3_image_size,
        token_mode=args.dinov3_token_mode,
        pool_size=args.dinov3_pool_size,
        dtype=getattr(torch, args.dinov3_dtype),
    )
    dinov3_strategy = anima_dinov3.AnimaDinoV3CachingStrategy(
        dinov3_encoder,
        cache_to_disk=args.cache_text_encoder_outputs_to_disk,
        batch_size=args.dinov3_batch_size,
        skip_disk_cache_validity_check=args.skip_cache_check,
    )

    dinov3_encoder.to(accelerator.device)
    anima_dinov3.cache_dinov3_embeddings(train_dataset_group, dinov3_strategy, accelerator)

    # The encoder is only needed again to embed reference images for sample prompts; keep it on CPU in that case.
    keep_dinov3_encoder = args.sample_prompts is not None
    dinov3_encoder.to("cpu")
    if not keep_dinov3_encoder:
        del dinov3_encoder.model
    clean_memory_on_device(accelerator.device)

    # The dataset reaches the DINOv3 cache through the text encoder caching strategy, and DataLoader workers get a
    # copy of it. Drop the encoder reference now that caching is done, so the ViT is not pickled into every worker.
    dinov3_strategy.encoder = None

    # Load tokenizers and set strategies
    logger.info("Loading tokenizers...")
    qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # Text encoder is always frozen for Anima
    qwen3_text_encoder.to(weight_dtype)
    qwen3_text_encoder.requires_grad_(False)

    # Cache text encoder outputs (DINOv3 embeddings are appended to them by the strategy below)
    qwen3_text_encoder.to(accelerator.device)
    qwen3_text_encoder.eval()

    text_encoder_caching_strategy = anima_dinov3.AnimaDinoV3TextEncoderOutputsCachingStrategy(
        args.cache_text_encoder_outputs_to_disk,
        args.text_encoder_batch_size,
        args.skip_cache_check,
        is_partial=False,
        dinov3_strategy=dinov3_strategy,
    )
    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

    with accelerator.autocast():
        train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)

    sample_prompts_te_outputs = None
    if args.sample_prompts is not None:
        logger.info(f"Cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
        prompts = sampling.load_prompts(args.sample_prompts)
        sample_prompts_te_outputs = {}
        with accelerator.autocast(), torch.no_grad():
            for prompt_dict in prompts:
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                    if p not in sample_prompts_te_outputs:
                        logger.info(f"  cache TE outputs for: {p}")
                        tokens_and_masks = tokenize_strategy.tokenize(p)
                        sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], tokens_and_masks
                        )

    accelerator.wait_for_everyone()

    qwen3_text_encoder = None
    gc.collect()
    clean_memory_on_device(accelerator.device)

    # Load VAE and cache latents
    logger.info("Loading Anima VAE...")
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)

    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # Load DiT (MiniTrainDIT + optional LLM Adapter) and wrap it with the DINOv3 projector
    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_model(
        "cpu", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, "cpu", dit_weight_dtype=None
    )

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(
            cpu_offload=args.cpu_offload_checkpointing,
            unsloth_offload=args.unsloth_offload_checkpointing,
        )

    model = anima_dinov3.create_dinov3_model(dit, dinov3_encoder, args)

    train_dit = args.learning_rate != 0
    projector_lr = args.dinov3_projector_lr if args.dinov3_projector_lr is not None else args.learning_rate
    train_projector = projector_lr != 0
    assert train_dit or train_projector, "nothing to train: both --learning_rate and --dinov3_projector_lr are 0"

    dit.requires_grad_(train_dit)
    model.dinov3_projector.requires_grad_(train_projector)

    # Block swap
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        logger.info(f"Enable block swap: blocks_to_swap={args.blocks_to_swap}")
        dit.enable_block_swap(args.blocks_to_swap, accelerator.device)

    # DiT offload to a second GPU (model parallelism)
    is_offloading_dit = args.dit_offload_blocks is not None and args.dit_offload_blocks > 0
    if is_offloading_dit:
        offload_device = torch.device(args.dit_offload_device)
        logger.info(f"Enable DiT offload: dit_offload_blocks={args.dit_offload_blocks}, device={offload_device}")
        dit.enable_dit_offload(args.dit_offload_blocks, accelerator.device, offload_device)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    # Setup optimizer with parameter groups
    param_groups = []
    param_group_names = []
    if train_dit:
        param_groups = anima_train_utils.get_anima_param_groups(
            dit,
            base_lr=args.learning_rate,
            self_attn_lr=args.self_attn_lr,
            cross_attn_lr=args.cross_attn_lr,
            mlp_lr=args.mlp_lr,
            mod_lr=args.mod_lr,
            llm_adapter_lr=args.llm_adapter_lr,
        )
        param_group_names = ["base", "self_attn", "cross_attn", "mlp", "mod", "llm_adapter"]
    if train_projector:
        param_groups.append({"params": list(model.dinov3_projector.parameters()), "lr": projector_lr})
        param_group_names.append("dinov3_projector")
        logger.info(f"  dinov3_projector params (lr={projector_lr})")

    training_models = [model]

    n_params = 0
    for group in param_groups:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train dit: {train_dit}, train dinov3 projector: {train_projector}")
    accelerator.print(f"number of trainable parameters: {n_params:,}")

    # prepare optimizer
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = optimizer_util.get_optimizer(args, trainable_params=param_groups)
    optimizer_train_fn, optimizer_eval_fn = optimizer_util.get_optimizer_train_eval_fn(optimizer, args)

    # prepare dataloader
    train_dataset_group.set_current_strategies()

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # calculate training steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # full fp16/bf16 training
    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed_precision='fp16'"
        accelerator.print("enable full fp16 training.")
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
    else:
        dit_weight_dtype = torch.float32  # If neither full_fp16 nor full_bf16, the model weights should be in float32
    model.to(dit_weight_dtype)  # DiT and projector

    clean_memory_on_device(accelerator.device)

    # Let us manage device placement when block-swap or DiT offload is active, since both split the DiT across devices.
    manual_placement = is_swapping_blocks or is_offloading_dit
    model = accelerator.prepare(model, device_placement=[not manual_placement])
    if manual_placement:
        unwrapped = accelerator.unwrap_model(model)
        if is_swapping_blocks:
            unwrapped.dit.move_to_device_except_swap_blocks(accelerator.device)
        else:
            unwrapped.dit.move_to_device_with_dit_offload(accelerator.device)
        unwrapped.dinov3_projector.to(accelerator.device)
    training_models = [model]
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if not cache_latents and vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    if args.full_fp16:
        accelerator_setup.patch_accelerator_for_fp16_training(accelerator)

    # resume
    args_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)

        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def create_grad_hook(p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_group))

    # Sampling: embed the prompt's reference image with DINOv3, if the prompt provides one.
    # `--ci <path>` / `image` in the prompt file selects the reference; without one the learned null condition is used.
    def on_prompt_start(prompt_dict, accel):
        unwrapped = accel.unwrap_model(model)
        image_path = prompt_dict.get("controlnet_image") or prompt_dict.get("image")
        if image_path is None or not keep_dinov3_encoder:
            unwrapped.set_sample_dinov3_embeds(None)  # null condition
            return
        from PIL import Image

        dinov3_encoder.to(accel.device)
        embeds = dinov3_encoder.encode_images([Image.open(image_path)])
        dinov3_encoder.to("cpu")
        clean_memory_on_device(accel.device)
        unwrapped.set_sample_dinov3_embeds(embeds.to(accel.device, dtype=unwrapped.dtype))

    def on_prompt_end(prompt_dict):
        accelerator.unwrap_model(model).set_sample_dinov3_embeds(None)

    def do_sample_images(epoch, global_step):
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            model,  # the wrapper is DiT-compatible: it appends the DINOv3 context inside forward
            vae,
            None,  # text encoder was freed; sample prompts are pre-encoded
            tokenize_strategy,
            text_encoding_strategy,
            sample_prompts_te_outputs,
            on_prompt_start=on_prompt_start,
            on_prompt_end=on_prompt_end,
        )

    # Training loop
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=args_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            import wandb

            wandb.define_metric("epoch")
            wandb.define_metric("loss/epoch", step_metric="epoch")

    if is_swapping_blocks:
        accelerator.unwrap_model(model).dit.prepare_block_swap_before_forward()

    # For --sample_at_first
    optimizer_eval_fn()
    do_sample_images(0, global_step)
    optimizer_train_fn()
    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    loss_recorder = logging_util.LossRecorder()
    epoch = 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(*training_models):
                # Get latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    if latents.ndim == 5:  # Fallback for 5D latents (old cache)
                        latents = latents.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
                else:
                    with torch.no_grad():
                        # images are already [-1, 1] from IMAGE_TRANSFORMS
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=dit_weight_dtype)

                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # Cached conditions:
                # [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate, dinov3_embeds]
                text_encoder_outputs_list = batch["text_encoder_outputs_list"]
                dinov3_embeds = text_encoder_outputs_list[5]
                caption_dropout_rates = text_encoder_outputs_list[4]

                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.drop_cached_text_encoder_outputs(
                    *text_encoder_outputs_list[:4], caption_dropout_rates=caption_dropout_rates
                )

                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit_weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)
                dinov3_embeds = dinov3_embeds.to(accelerator.device, dtype=dit_weight_dtype)

                # Noise and timesteps
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents, noise, accelerator.device, dit_weight_dtype
                )
                timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

                if torch.any(torch.isnan(noisy_model_input)):
                    accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                    noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                # padding_mask: (B, 1, H_latent, W_latent)
                bs = latents.shape[0]
                padding_mask = torch.zeros(
                    bs, 1, latents.shape[-2], latents.shape[-1], dtype=dit_weight_dtype, device=accelerator.device
                )

                # DiT forward. The LLM adapter and the DINOv3 projector run inside the wrapper's forward,
                # so DDP synchronizes their gradients.
                noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, (B, C, 1, H, W)
                with accelerator.autocast():
                    model_pred = model(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        padding_mask=padding_mask,
                        source_attention_mask=attn_mask,
                        t5_input_ids=t5_input_ids,
                        t5_attn_mask=t5_attn_mask,
                        dinov3_embeds=dinov3_embeds,
                    )
                model_pred = model_pred.squeeze(2)  # 5D to 4D, (B, C, H, W)

                # Compute loss (rectified flow: target = noise - latents)
                target = noise - latents

                weighting = anima_train_utils.compute_loss_weighting_for_anima(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps, None)
                loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])  # (B, C, H, W) -> (B,)

                if weighting is not None:
                    loss = loss * weighting

                loss = loss * batch["loss_weights"]
                loss = loss.mean()

                accelerator.backward(loss)

                if not args.fused_backward_pass:
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                do_sample_images(None, global_step)

                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            unwrapped.dit if train_dit else None,
                        )
                        if train_projector:
                            save_dinov3_projector(args, unwrapped, dinov3_encoder, global_step=global_step)
                optimizer_train_fn()

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                optimizer_util.append_lr_to_logs_with_names(logs, lr_scheduler, args.optimizer_type, param_group_names)
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            progress_bar.set_postfix(**{"avr_loss": loss_recorder.moving_average})

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            accelerator.log({"loss/epoch": loss_recorder.moving_average, "epoch": epoch + 1}, step=global_step)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    unwrapped.dit if train_dit else None,
                )
                if train_projector:
                    save_dinov3_projector(args, unwrapped, dinov3_encoder, epoch=epoch + 1)

        do_sample_images(epoch + 1, global_step)

    # End training
    is_main_process = accelerator.is_main_process
    model = accelerator.unwrap_model(model)

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        checkpoint_io.save_state_on_train_end(args, accelerator)

    del accelerator

    if is_main_process:
        if train_dit:
            anima_train_utils.save_anima_model_on_train_end(args, save_dtype, epoch, global_step, model.dit)
        if train_projector:
            save_dinov3_projector(args, model, dinov3_encoder)
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    args_util.add_sd_models_arguments(parser)
    args_util.add_dataset_arguments(parser, True, True, True)
    args_util.add_training_arguments(parser, False)
    args_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    args_util.add_sd_saving_arguments(parser)
    args_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    anima_dinov3.add_dinov3_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="offload gradient checkpointing to CPU (reduces VRAM at cost of speed)",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    if args.show_timesteps:
        anima_train_utils.show_timesteps(args)
    else:
        train(args)
