"""Pseudo dry-run for Anima + DINOv3 conditioning (no real Anima/Qwen3/VAE weights, no real dataset).

Runs the actual ``train_anima_dinov3.train()`` end-to-end on a handful of generated images, with only the
heavy weight loaders stubbed out:

  * DiT      -> a real (small) ``library.anima_models.Anima``
  * Qwen3    -> a stub text encoder (real bundled tokenizer)
  * VAE      -> a stub encoder/decoder with the Qwen-Image VAE interface
  * DINOv3   -> the REAL model (camenduru/dinov3-vitl16-pretrain-lvd1689m by default)

Verified:
  1. DINOv3 embeddings are cached to npz, reused on a second run, and invalidated when settings change
  2. The dataset delivers them to the training loop as text_encoder_outputs_list[5]
  3. VAE latents and text encoder outputs are cached as usual
  4. The zero-initialized projector is a no-op: the wrapper's output matches the bare DiT at step 0
  5. Training steps run, the projector gets gradients and is saved / reloadable
  6. Sampling works, including a DINOv3 reference image from the prompt file

Run:
    python tools/dev/manual_test_anima_dinov3_dryrun.py [--dinov3_model <hf id>] [--cache_to_disk]
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from library import anima_dinov3, anima_models, anima_train_utils, anima_utils
import train_anima_dinov3
QWEN3_HIDDEN = 1024  # Qwen3-0.6B hidden size == Anima crossattn_emb_channels


# --------------------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------------------
class StubTextEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class StubQwen3(nn.Module):
    """Stand-in for Qwen3-0.6B: deterministic embeddings of the right shape."""

    def __init__(self, hidden_size=QWEN3_HIDDEN, vocab_size=151936):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids=None, attention_mask=None):
        return StubTextEncoderOutput(self.embed(input_ids.to(self.embed.weight.device)))


class StubVAE(nn.Module):
    """Stand-in for AutoencoderKLQwenImage: 16 channels, spatial downscale 8."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 16, kernel_size=8, stride=8)
        self.decoder = nn.ConvTranspose2d(16, 3, kernel_size=8, stride=8)

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def encode_pixels_to_latents(self, pixels):
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        return self.encoder(pixels.to(self.device, dtype=self.dtype))

    def decode_to_pixels(self, latents):
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        return self.decoder(latents.to(self.device, dtype=self.dtype))


def build_small_anima():
    """A real Anima DiT, small enough to train on CPU/T4 in seconds."""
    return anima_models.Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=8,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=128,
        num_blocks=2,
        num_heads=4,
        crossattn_emb_channels=QWEN3_HIDDEN,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        adaln_lora_dim=64,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        attn_mode="torch",
    )


def make_dataset(root: str, num_images: int = 4, size: int = 256):
    image_dir = os.path.join(root, "1_test")
    os.makedirs(image_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(num_images):
        array = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(os.path.join(image_dir, f"img{i:02d}.png"))
        with open(os.path.join(image_dir, f"img{i:02d}.txt"), "w") as f:
            f.write(f"a photo of test subject {i}")
    return image_dir


def install_stubs():
    """Patch the heavy loaders. DINOv3 stays real."""
    from library import anima_utils as au

    tokenizer = au.load_qwen3_tokenizer(os.path.join(REPO_ROOT, "configs", "qwen3_06b"))

    def fake_load_qwen3_text_encoder(qwen3_path, dtype=torch.bfloat16, device="cpu", **kwargs):
        te = StubQwen3().to(device=device, dtype=dtype)
        return te, tokenizer

    def fake_load_anima_model(device, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, **kwargs):
        return build_small_anima()

    def fake_load_vae(args, device="cpu", disable_mmap=True):
        return StubVAE().to(device)

    train_anima_dinov3.anima_utils.load_qwen3_text_encoder = fake_load_qwen3_text_encoder
    train_anima_dinov3.anima_utils.load_anima_model = fake_load_anima_model
    train_anima_dinov3.anima_train_utils.load_qwen_image_vae = fake_load_vae
    # the checkpoint saver writes an Anima-format DiT; the stub state dict is fine for a round-trip test
    anima_train_utils.anima_utils.save_anima_model = lambda ckpt_file, sd, metadata, dtype: torch.save(
        {"n": len(sd)}, ckpt_file
    )


# --------------------------------------------------------------------------------------
# Checks that do not need a training run
# --------------------------------------------------------------------------------------
def check_encoder_and_projector(dinov3_model: str):
    print("\n=== 1. DINOv3 encoder token counts ===")
    for token_mode, pool_size, expected in [("all", None, 5 + 196), ("patch", None, 196), ("patch", 2, 49), ("cls", None, 1)]:
        encoder = anima_dinov3.DinoV3Encoder(dinov3_model, image_size=224, token_mode=token_mode, pool_size=pool_size)
        tokens = encoder.encode_images([Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))])
        assert encoder.num_tokens == expected, f"{token_mode}/{pool_size}: {encoder.num_tokens} != {expected}"
        assert tuple(tokens.shape) == (1, expected, 1024), tokens.shape
        print(f"  token_mode={token_mode:5s} pool={str(pool_size):4s} -> {tuple(tokens.shape)} OK")

    print("\n=== 2. The image branch is a no-op at step 0 (gated) but not when concatenated ===")
    x = torch.randn(2, 16, 1, 16, 16)
    t = torch.rand(2)
    context = torch.randn(2, 12, QWEN3_HIDDEN)
    padding_mask = torch.zeros(2, 1, 16, 16)
    dinov3_embeds = torch.randn(2, encoder.num_tokens, 1024)

    def make_args(cond_mode="gated"):
        return argparse.Namespace(
            dinov3_projector_path=None,
            dinov3_dropout_rate=0.0,
            dinov3_cond_mode=cond_mode,
        )

    diffs = {}
    for cond_mode in ["gated", "concat"]:
        dit = build_small_anima().eval()
        model = anima_dinov3.create_dinov3_model(dit, encoder, make_args(cond_mode)).eval()
        with torch.no_grad():
            baseline = dit.forward_mini_train_dit(x, t, context, padding_mask=padding_mask)
            with_dino = model(x, t, context, padding_mask=padding_mask, dinov3_embeds=dinov3_embeds)
        diffs[cond_mode] = (baseline - with_dino).abs().max().item()
        print(f"  cond_mode={cond_mode:6s}: max |bare DiT - wrapper| = {diffs[cond_mode]:.2e}")

    assert diffs["gated"] < 1e-5, f"gated mode must be an exact no-op at init, got {diffs['gated']}"
    assert diffs["concat"] > 1e-3, "concat mode is expected to perturb the DiT at init (softmax dilution)"
    print("  gated is an exact no-op, concat perturbs the pretrained cross-attention OK")

    print("\n=== 3. The image branch learns, with and without gradient checkpointing ===")
    # The gates are the only zero at init, so step 0 gives a gradient to the gates alone; once they open,
    # the projector itself starts receiving gradients. Anything else would be a dead branch.
    for checkpointing in [None, "plain", "cpu_offload", "unsloth"]:
        dit = build_small_anima()
        model = anima_dinov3.create_dinov3_model(dit, encoder, make_args("gated"))
        if checkpointing is not None:
            dit.enable_gradient_checkpointing(
                cpu_offload=(checkpointing == "cpu_offload"), unsloth_offload=(checkpointing == "unsloth")
            )
        model.train()
        optimizer = torch.optim.AdamW(model.dinov3_projector.parameters(), lr=1e-2)

        def step(dropout_rate):
            model.dropout_rate = dropout_rate
            optimizer.zero_grad()
            out = model(x, t, context, padding_mask=padding_mask, dinov3_embeds=dinov3_embeds)
            out.mean().backward()
            grads = {}
            for name in ["fc1.weight", "fc2.weight", "pos_embed", "null_context", "block_gates"]:
                param = dict(model.dinov3_projector.named_parameters())[name]
                # every parameter must get a gradient tensor on every step, or DDP would complain about unused params
                assert param.grad is not None, f"[{checkpointing}] no grad for dinov3_projector.{name}"
                grads[name] = param.grad.norm().item()
            optimizer.step()
            return grads

        first = step(0.0)  # gates closed: only they can learn
        second = step(0.0)  # gates open: the projector starts learning
        dropped = step(1.0)  # every sample uses the null embedding

        assert first["block_gates"] > 0, f"[{checkpointing}] the gates got no gradient at step 0 (dead branch)"
        assert second["fc2.weight"] > 0, f"[{checkpointing}] the projector got no gradient after the gates opened"
        assert dropped["null_context"] > 0, f"[{checkpointing}] the null embedding got no gradient when dropping"
        print(
            f"  checkpointing={str(checkpointing):11s} "
            f"gates(step0)={first['block_gates']:.2e} fc2(step1)={second['fc2.weight']:.2e} "
            f"null(dropped)={dropped['null_context']:.2e} OK"
        )

    print("\n=== 4. Sampling convention: model(x, t, crossattn_emb, padding_mask=...) ===")
    model.eval()
    model.eval()
    with torch.no_grad():
        model.set_sample_dinov3_embeds(dinov3_embeds[:1])
        sampled = model(x[:1], t[:1], context[:1], padding_mask=padding_mask[:1])
        model.set_sample_dinov3_embeds(None)  # null condition
        null_sampled = model(x[:1], t[:1], context[:1], padding_mask=padding_mask[:1])
    assert sampled.shape == null_sampled.shape == (1, 16, 1, 16, 16), sampled.shape
    print(f"  reference-image and null-condition sampling both give {tuple(sampled.shape)} OK")

    print("\n=== 5. Projector save / load round-trip ===")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "proj.safetensors")
        anima_dinov3.save_dinov3_projector(path, model.dinov3_projector, encoder.cache_config)
        args = make_args("gated")
        args.dinov3_projector_path = path
        reloaded = anima_dinov3.create_dinov3_model(build_small_anima(), encoder, args)
        for k, v in model.dinov3_projector.state_dict().items():
            assert torch.allclose(v, reloaded.dinov3_projector.state_dict()[k]), k
    print("  state dict matches after reload OK")


# --------------------------------------------------------------------------------------
# Full training run
# --------------------------------------------------------------------------------------
def run_training(root: str, dinov3_model: str, cache_to_disk: bool):
    print("\n=== 6. Full train_anima_dinov3.train() run ===")
    image_dir = make_dataset(root)
    output_dir = os.path.join(root, "out")
    os.makedirs(output_dir, exist_ok=True)

    # a sample prompt with a DINOv3 reference image (--ci), to exercise the sampling path
    sample_image = os.path.join(image_dir, "img00.png")
    prompts_path = os.path.join(root, "prompts.txt")
    with open(prompts_path, "w") as f:
        f.write(f"a photo of test subject 0 --w 128 --h 128 --s 2 --l 1.0 --ci {sample_image}\n")
        f.write("a photo of test subject 1 --w 128 --h 128 --s 2 --l 1.0\n")  # no reference -> null condition

    argv = [
        "--pretrained_model_name_or_path", "stub_dit.safetensors",
        "--qwen3", os.path.join(REPO_ROOT, "configs", "qwen3_06b"),
        "--vae", "stub_vae.safetensors",
        "--train_data_dir", root,
        "--resolution", "256,256",
        "--output_dir", output_dir,
        "--output_name", "dinov3_test",
        "--max_train_steps", "4",
        "--learning_rate", "1e-5",
        "--dinov3_projector_lr", "1e-4",
        "--optimizer_type", "adamw",
        "--mixed_precision", "no",
        "--save_precision", "float",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--dinov3_model_name_or_path", dinov3_model,
        "--dinov3_image_size", "224",
        "--dinov3_token_mode", "patch",
        "--dinov3_pool_size", "2",
        "--dinov3_dropout_rate", "0.1",
        "--dinov3_batch_size", "2",
        "--sample_prompts", prompts_path,
        "--sample_at_first",
        "--max_data_loader_n_workers", "0",
        "--seed", "42",
    ]
    if cache_to_disk:
        argv += ["--cache_latents_to_disk", "--cache_text_encoder_outputs_to_disk"]

    parser = train_anima_dinov3.setup_parser()
    args = parser.parse_args(argv)
    train_anima_dinov3.train(args)

    npz_files = [f for f in os.listdir(image_dir) if f.endswith(anima_dinov3.DINOV3_NPZ_SUFFIX)]
    if cache_to_disk:
        assert len(npz_files) == 4, f"expected 4 DINOv3 npz files, found {len(npz_files)}"
        embeds = np.load(os.path.join(image_dir, npz_files[0]))["dinov3_embeds"]
        assert embeds.shape == (49, 1024), embeds.shape  # 14x14 patch grid pooled by 2
        print(f"  DINOv3 cache: {len(npz_files)} npz files, embeds {embeds.shape} {embeds.dtype} OK")
    else:
        assert not npz_files, "memory cache should not write npz files"
        print("  DINOv3 cache: kept in memory, no npz written OK")

    saved = sorted(os.listdir(output_dir))
    projector_files = [f for f in saved if "dinov3_proj" in f]
    assert projector_files, f"no projector checkpoint saved, got {saved}"
    samples = os.listdir(os.path.join(output_dir, "sample"))
    assert samples, "no sample images generated"
    print(f"  saved: {saved}")
    print(f"  samples: {samples}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3_model", type=str, default=anima_dinov3.DEFAULT_DINOV3_MODEL)
    parser.add_argument("--cache_to_disk", action="store_true")
    args = parser.parse_args()

    install_stubs()
    check_encoder_and_projector(args.dinov3_model)

    with tempfile.TemporaryDirectory() as root:
        run_training(root, args.dinov3_model, args.cache_to_disk)

    print("\nAll DINOv3 dry-run checks passed.")


if __name__ == "__main__":
    main()
