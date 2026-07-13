# DINOv3 image conditioning for Anima training.
#
# Adds DINOv3 (https://huggingface.co/docs/transformers/en/model_doc/dinov3) embeddings as a second conditioning
# signal next to the Qwen3 text encoder. A trainable projector maps the DINOv3 tokens into the DiT cross-attention
# context space; each block then attends to them in a decoupled cross-attention added through a zero-initialized
# gate (`--dinov3_cond_mode gated`, the default), or, optionally, they are concatenated to the text context
# (`--dinov3_cond_mode concat`).
#
# Caching: DINOv3 embeddings are cached per image (disk or memory) and delivered to the training loop through the
# existing text-encoder-outputs channel, so no dataset changes are needed.

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from library import strategy_anima
from library.device_utils import clean_memory_on_device
from library.strategy_base import TextEncodingStrategy, TokenizeStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


DEFAULT_DINOV3_MODEL = "camenduru/dinov3-vitl16-pretrain-lvd1689m"  # mirror of facebook/dinov3-vitl16-pretrain-lvd1689m
DINOV3_NPZ_SUFFIX = "_dinov3.npz"

# ImageNet statistics, matching the DINOv3 image processor defaults
DINOV3_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGE_STD = (0.229, 0.224, 0.225)


def add_dinov3_arguments(parser: argparse.ArgumentParser):
    """DINOv3 conditioning arguments."""
    parser.add_argument(
        "--dinov3_model_name_or_path",
        type=str,
        default=DEFAULT_DINOV3_MODEL,
        help="DINOv3 model (HF repo id or local dir). Default: mirror of the 0.3B ViT-L/16 model",
    )
    parser.add_argument(
        "--dinov3_image_size",
        type=int,
        default=224,
        help="Square resolution the image is resized to before DINOv3. Must be a multiple of the patch size (16)",
    )
    parser.add_argument(
        "--dinov3_token_mode",
        type=str,
        default="all",
        choices=["all", "patch", "cls"],
        help="Which DINOv3 tokens to condition on: all (CLS + registers + patches), patch (patches only), cls (CLS only)",
    )
    parser.add_argument(
        "--dinov3_pool_size",
        type=int,
        default=None,
        help="Average-pool the patch token grid by this factor to reduce token count (e.g. 2: 14x14 -> 7x7)",
    )
    parser.add_argument(
        "--dinov3_cond_mode",
        type=str,
        default="gated",
        choices=["gated", "concat"],
        help="How the DINOv3 tokens enter the DiT. gated: a decoupled cross-attention per block behind a zero-init "
        "gate (exact no-op at step 0, recommended). concat: appended to the text context (cheaper, but perturbs the "
        "pretrained cross-attention from step 0)",
    )
    parser.add_argument(
        "--dinov3_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Compute dtype for the DINOv3 encoder. float16 overflows on DINOv3's activation outliers and produces "
        "NaN embeddings, so it is NOT the default even under --mixed_precision fp16",
    )
    parser.add_argument(
        "--dinov3_batch_size",
        type=int,
        default=8,
        help="Batch size for caching DINOv3 embeddings",
    )
    parser.add_argument(
        "--dinov3_dropout_rate",
        type=float,
        default=0.0,
        help="Probability of replacing the DINOv3 condition with the learned null embedding (enables image-free CFG)",
    )
    parser.add_argument(
        "--dinov3_projector_lr",
        type=float,
        default=None,
        help="Learning rate for the DINOv3 projector. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--dinov3_projector_path",
        type=str,
        default=None,
        help="Path to previously saved DINOv3 projector weights (safetensors) to resume from",
    )
    return parser


class DinoV3Encoder(nn.Module):
    """Wraps the HF DINOv3 model: preprocessing, forward, token selection and pooling."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_DINOV3_MODEL,
        image_size: int = 224,
        token_mode: str = "all",
        pool_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        logger.info(f"Loading DINOv3 model: {model_name_or_path}")
        try:
            model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype)
        except TypeError:  # transformers < 5
            model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        model.requires_grad_(False)
        model.eval()

        self.model = model
        self.model_name_or_path = model_name_or_path
        self.image_size = image_size
        self.token_mode = token_mode
        self.pool_size = pool_size

        config = model.config
        self.hidden_size: int = config.hidden_size
        self.patch_size: int = config.patch_size
        # DINOv3 prepends 1 CLS token plus `num_register_tokens` register tokens to the patch tokens
        self.num_special_tokens: int = 1 + int(getattr(config, "num_register_tokens", 0) or 0)

        if image_size % self.patch_size != 0:
            raise ValueError(f"--dinov3_image_size ({image_size}) must be a multiple of the patch size ({self.patch_size})")
        self.grid_size = image_size // self.patch_size

        if pool_size is not None and pool_size > 1:
            if self.grid_size % pool_size != 0:
                raise ValueError(
                    f"--dinov3_pool_size ({pool_size}) must divide the patch grid size "
                    f"({self.grid_size} = {image_size} / {self.patch_size})"
                )

        logger.info(
            f"DINOv3: hidden_size={self.hidden_size}, patch_size={self.patch_size}, grid={self.grid_size}x{self.grid_size}, "
            f"token_mode={token_mode}, pool_size={pool_size}, num_tokens={self.num_tokens}"
        )

    @property
    def num_tokens(self) -> int:
        """Number of conditioning tokens produced per image."""
        if self.token_mode == "cls":
            return 1
        pooled_grid = self.grid_size // self.pool_size if self.pool_size and self.pool_size > 1 else self.grid_size
        num_patch_tokens = pooled_grid * pooled_grid
        if self.token_mode == "patch":
            return num_patch_tokens
        return self.num_special_tokens + num_patch_tokens

    @property
    def cache_config(self) -> Dict[str, Any]:
        """Identifies the embeddings on disk: a cache written with different settings is stale."""
        return {
            "model": self.model_name_or_path,
            "image_size": self.image_size,
            "token_mode": self.token_mode,
            "pool_size": self.pool_size or 1,
            "hidden_size": self.hidden_size,
            "num_tokens": self.num_tokens,
        }

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalized (3, S, S) float tensor."""
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(DINOV3_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(DINOV3_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
        return (pixels - mean) / std

    def preprocess_paths(self, paths: List[str]) -> torch.Tensor:
        return torch.stack([self.preprocess(Image.open(p)) for p in paths])

    def _select_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, num_special + grid*grid, D) -> (B, num_tokens, D)"""
        if self.token_mode == "cls":
            return hidden_states[:, :1]

        special = hidden_states[:, : self.num_special_tokens]
        patches = hidden_states[:, self.num_special_tokens :]

        if self.pool_size is not None and self.pool_size > 1:
            b, n, d = patches.shape
            grid = self.grid_size
            patches = patches.reshape(b, grid, grid, d).permute(0, 3, 1, 2)  # (B, D, grid, grid)
            patches = torch.nn.functional.avg_pool2d(patches, self.pool_size)
            patches = patches.permute(0, 2, 3, 1).reshape(b, -1, d)

        if self.token_mode == "patch":
            return patches
        return torch.cat([special, patches], dim=1)

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Normalized pixels (B, 3, S, S) -> conditioning tokens (B, num_tokens, D), float32 on CPU."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        outputs = self.model(pixel_values=pixel_values.to(device, dtype=dtype))
        tokens = self._select_tokens(outputs.last_hidden_state).float().cpu()

        # DINOv3 has large activation outliers: in float16 they overflow and the whole cache silently becomes NaN
        if not torch.isfinite(tokens).all():
            raise RuntimeError(
                f"DINOv3 produced non-finite embeddings in {dtype}. Use --dinov3_dtype float32 (or bfloat16)."
            )
        return tokens

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        return self.encode(torch.stack([self.preprocess(img) for img in images]))


class AnimaDinoV3CachingStrategy:
    """Caches DINOv3 embeddings to `<image>_dinov3.npz` (or onto the ImageInfo when caching to memory)."""

    def __init__(self, encoder: DinoV3Encoder, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool):
        self.encoder = encoder
        self.cache_to_disk = cache_to_disk
        self.batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check

    @staticmethod
    def get_embeds_npz_path(image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + DINOV3_NPZ_SUFFIX

    def is_disk_cached_embeds_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            npz = np.load(npz_path)
            if "dinov3_embeds" not in npz or "cache_config" not in npz:
                return False
            cached_config = json.loads(str(npz["cache_config"]))
            if cached_config != self.encoder.cache_config:
                # settings changed (model, resolution, token mode...): the cache is stale and gets rewritten
                return False
            if npz["dinov3_embeds"].shape[0] != self.encoder.num_tokens:
                return False
        except Exception:
            logger.warning(f"Failed to read DINOv3 cache, regenerating: {npz_path}")
            return False
        return True

    def load_embeds_npz(self, npz_path: str) -> np.ndarray:
        return np.load(npz_path)["dinov3_embeds"]

    def cache_batch_embeds(self, infos: List) -> None:
        embeds = self.encoder.encode_images([Image.open(info.absolute_path) for info in infos])
        config_array = np.array(json.dumps(self.encoder.cache_config))

        for i, info in enumerate(infos):
            embeds_i = embeds[i].numpy()
            if self.cache_to_disk:
                # fp16 halves the cache size; the projector's input LayerNorm makes the precision loss irrelevant
                np.savez(
                    self.get_embeds_npz_path(info.absolute_path),
                    dinov3_embeds=embeds_i.astype(np.float16),
                    cache_config=config_array,
                )
            else:
                info.dinov3_embeds = embeds_i


def cache_dinov3_embeddings(dataset_group, strategy: AnimaDinoV3CachingStrategy, accelerator: Accelerator) -> None:
    """Run the DINOv3 encoder over every image in the dataset. Must run before caching text encoder outputs."""
    for dataset in dataset_group.datasets:
        infos = list(dataset.image_data.values())

        for info in infos:
            subset = dataset.image_to_subset[info.image_key]
            if subset.flip_aug and strategy.encoder.token_mode != "cls":
                raise ValueError(
                    "flip_aug is not supported with spatial DINOv3 tokens: the cached embeddings describe the "
                    "unflipped image, so they would not line up with a flipped latent. "
                    "Use --dinov3_token_mode cls or disable flip_aug."
                )

        # sharding across processes is only valid on disk, where every process reads back the whole cache
        if strategy.cache_to_disk and accelerator.num_processes > 1:
            infos = infos[accelerator.process_index :: accelerator.num_processes]

        todo = []
        for info in infos:
            npz_path = strategy.get_embeds_npz_path(info.absolute_path)
            if strategy.is_disk_cached_embeds_expected(npz_path):
                continue
            todo.append(info)

        logger.info(f"Caching DINOv3 embeddings for {len(todo)} images (skipped {len(infos) - len(todo)} already cached)")

        for i in tqdm(range(0, len(todo), strategy.batch_size), desc="DINOv3 embeddings"):
            strategy.cache_batch_embeds(todo[i : i + strategy.batch_size])

    accelerator.wait_for_everyone()


class AnimaDinoV3TextEncoderOutputsCachingStrategy(strategy_anima.AnimaTextEncoderOutputsCachingStrategy):
    """Anima text encoder cache plus the DINOv3 embeddings as a 6th element.

    The dataset stacks whatever `load_outputs_npz` returns into `batch["text_encoder_outputs_list"]`, so appending
    the DINOv3 embeddings here is all that is needed to get them into the training loop:

        [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate, dinov3_embeds]

    DINOv3 embeddings live in their own npz next to the image, so swapping the DINOv3 model does not invalidate
    the text encoder cache (and vice versa).
    """

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        dinov3_strategy: Optional[AnimaDinoV3CachingStrategy] = None,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.dinov3_strategy = dinov3_strategy

    def get_dinov3_npz_path(self, te_npz_path: str) -> str:
        base = te_npz_path[: -len(self.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX)]
        return base + DINOV3_NPZ_SUFFIX

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        outputs = super().load_outputs_npz(npz_path)
        dinov3_npz_path = self.get_dinov3_npz_path(npz_path)
        if not os.path.exists(dinov3_npz_path):
            raise FileNotFoundError(
                f"DINOv3 embeddings not found: {dinov3_npz_path}. The DINOv3 cache must be built before training."
            )
        outputs.append(self.dinov3_strategy.load_embeds_npz(dinov3_npz_path))
        return outputs

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        super().cache_batch_outputs(tokenize_strategy, models, text_encoding_strategy, infos)

        if not self.cache_to_disk:
            # memory cache: the DINOv3 pass stored the embeddings on the ImageInfo, append them to the tuple
            for info in infos:
                dinov3_embeds = getattr(info, "dinov3_embeds", None)
                if dinov3_embeds is None:
                    raise RuntimeError(f"DINOv3 embeddings missing for {info.absolute_path}")
                info.text_encoder_outputs = tuple(list(info.text_encoder_outputs) + [dinov3_embeds])


class DinoV3Projector(nn.Module):
    """Projects DINOv3 tokens into the DiT cross-attention context space.

    Also owns the per-block gates for the decoupled cross-attention. Keeping them here (rather than inside the DiT)
    means the DiT state dict is unchanged and its checkpoints stay loadable by the stock Anima scripts: everything
    DINOv3-related lives in one small side file.
    """

    def __init__(self, in_dim: int, out_dim: int, num_tokens: int, num_blocks: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens = num_tokens

        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, out_dim))
        # condition used when the DINOv3 signal is dropped, and at sampling time when no reference image is given
        self.null_context = nn.Parameter(torch.randn(1, num_tokens, out_dim) * 0.02)
        # One gate per DiT block, zero-initialized: the image branch starts closed and is the *only* zero in the
        # branch. Zeroing the projector output as well would kill the gate's own gradient (its input would be zero),
        # leaving the branch permanently dead. The gate still receives a gradient at step 0, so it opens up.
        self.block_gates = nn.Parameter(torch.zeros(num_blocks))

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, dinov3_embeds: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(dinov3_embeds)
        x = self.fc2(self.act(self.fc1(x)))
        return x + self.pos_embed


class AnimaDinoV3Model(nn.Module):
    """DiT + DINOv3 projector as one module, so DDP sees every trainable parameter.

    Two ways to feed the image tokens into the DiT (`--dinov3_cond_mode`):

      * `gated` (default): each block runs a second, decoupled cross-attention over the DINOv3 tokens, added to the
        text cross-attention through a zero-initialized per-block gate (IP-Adapter style). At step 0 this is an exact
        no-op, so training starts from the pretrained model.
      * `concat`: the tokens are appended to the text context. Simpler and cheaper, but at step 0 they already take
        softmax mass away from the text tokens, i.e. the pretrained behaviour is perturbed before anything is learned.

    Two calling conventions:
      * training: pass raw `prompt_embeds` with `t5_input_ids`; the LLM adapter runs here (inside the DDP-wrapped
        forward, like the stock Anima DiT does internally) and `dinov3_embeds` come from the batch.
      * sampling: `library.anima_train_utils.do_sample` calls `model(x, t, crossattn_emb, padding_mask=...)` with the
        adapter already applied and no DINOv3 tensor; the embeddings set by `set_sample_dinov3_embeds` are used
        instead (falling back to the learned null condition).
    """

    def __init__(self, dit, projector: DinoV3Projector, dropout_rate: float = 0.0, cond_mode: str = "gated") -> None:
        super().__init__()
        self.dit = dit
        self.dinov3_projector = projector
        self.dropout_rate = dropout_rate
        self.cond_mode = cond_mode
        self._sample_dinov3_embeds: Optional[torch.Tensor] = None

    def dinov3_parameters(self):
        return list(self.dinov3_projector.parameters())

    def __getattr__(self, name: str):
        # delegate the DiT's own API (block swap, use_llm_adapter, llm_adapter, ...) to the wrapped model
        try:
            return super().__getattr__(name)
        except AttributeError:
            dit = self._modules.get("dit")
            if dit is None:
                raise
            return getattr(dit, name)

    @property
    def device(self):
        return self.dit.device

    @property
    def dtype(self):
        return self.dit.dtype

    def set_sample_dinov3_embeds(self, embeds: Optional[torch.Tensor]) -> None:
        self._sample_dinov3_embeds = embeds

    def _build_dinov3_context(self, dinov3_embeds: Optional[torch.Tensor], batch_size: int, dtype, device) -> torch.Tensor:
        null_context = self.dinov3_projector.null_context.to(dtype=dtype, device=device)

        if dinov3_embeds is None:
            return null_context.expand(batch_size, -1, -1)

        dinov3_embeds = dinov3_embeds.to(device=device, dtype=dtype)
        if dinov3_embeds.ndim == 2:
            dinov3_embeds = dinov3_embeds.unsqueeze(0)
        if dinov3_embeds.shape[0] == 1 and batch_size > 1:
            dinov3_embeds = dinov3_embeds.expand(batch_size, -1, -1)

        context = self.dinov3_projector(dinov3_embeds)

        # torch.where (rather than an `if`) keeps null_context in the autograd graph on every step, which DDP requires
        if self.training and self.dropout_rate > 0.0:
            drop = torch.rand(batch_size, device=device) < self.dropout_rate
        else:
            drop = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return torch.where(drop.view(-1, 1, 1), null_context, context)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
        dinov3_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if t5_input_ids is not None and self.dit.use_llm_adapter:
            context = self.dit.llm_adapter(
                source_hidden_states=context,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_attention_mask,
            )
            if t5_attn_mask is not None:
                context = context.masked_fill(~t5_attn_mask.bool().unsqueeze(-1), 0)

        if dinov3_embeds is None:
            dinov3_embeds = self._sample_dinov3_embeds

        dinov3_context = self._build_dinov3_context(dinov3_embeds, context.shape[0], context.dtype, context.device)

        if self.cond_mode == "concat":
            context = torch.cat([context, dinov3_context], dim=1)
            dinov3_context = None
            gates = None
        else:
            gates = self.dinov3_projector.block_gates.to(device=context.device)

        # the adapter already ran (or is unused), so do not let the DiT run it again
        return self.dit.forward_mini_train_dit(
            x,
            timesteps,
            context,
            fps=fps,
            padding_mask=padding_mask,
            dinov3_emb=dinov3_context,
            dinov3_gates=gates,
        )


def create_dinov3_model(dit, encoder: DinoV3Encoder, args: argparse.Namespace) -> AnimaDinoV3Model:
    context_dim = dit.blocks[0].cross_attn.context_dim
    projector = DinoV3Projector(
        in_dim=encoder.hidden_size,
        out_dim=context_dim,
        num_tokens=encoder.num_tokens,
        num_blocks=len(dit.blocks),
    )
    n_params = sum(p.numel() for p in projector.parameters())
    logger.info(
        f"DINOv3 projector: {encoder.hidden_size} -> {context_dim}, {encoder.num_tokens} tokens, {n_params:,} parameters, "
        f"cond_mode={args.dinov3_cond_mode}"
    )

    if args.dinov3_projector_path is not None:
        from safetensors.torch import load_file

        logger.info(f"Loading DINOv3 projector weights: {args.dinov3_projector_path}")
        projector.load_state_dict(load_file(args.dinov3_projector_path))

    return AnimaDinoV3Model(dit, projector, dropout_rate=args.dinov3_dropout_rate, cond_mode=args.dinov3_cond_mode)


def save_dinov3_projector(path: str, projector: DinoV3Projector, encoder_config: Dict[str, Any]) -> None:
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state_dict = {k: v.detach().clone().to("cpu", dtype=torch.float32) for k, v in projector.state_dict().items()}
    metadata = {k: str(v) for k, v in encoder_config.items()}
    save_file(state_dict, path, metadata=metadata)
    logger.info(f"Saved DINOv3 projector: {path}")
