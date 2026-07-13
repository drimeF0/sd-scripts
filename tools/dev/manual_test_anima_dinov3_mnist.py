"""Real (small-scale) training run of train_anima_dinov3.py on 1k MNIST digits, single GPU.

Trains a small randomly-initialized Anima DiT so the run is cheap; DINOv3 is the real model. The text encoder is a
stub (a frozen random embedding), so the *only* informative conditioning signal about the digit is the DINOv3
embedding of the source image. The VAE is a stub codec whose latent is literally the 8x-downsampled greyscale
image, which keeps the samples interpretable.

After training it measures whether the model actually uses the DINOv3 condition: the flow-matching loss is
evaluated with the matching embeddings and with shuffled (mismatched) ones, on identical noise and timesteps.
A model that ignores the condition scores the same either way.

Run:
    python tools/dev/manual_test_anima_dinov3_mnist.py [--max_train_steps 1500] [--batch_size 8]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from library import anima_dinov3, anima_models, anima_train_utils, logging_util
import train_anima_dinov3

QWEN3_HIDDEN = 1024
MNIST_CSV = "/content/sample_data/mnist_train_small.csv"


class StubTextEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class StubQwen3(nn.Module):
    """Frozen random embedding: carries the caption, but nothing pretrained."""

    def __init__(self, hidden_size=QWEN3_HIDDEN, vocab_size=151936):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids=None, attention_mask=None):
        return StubTextEncoderOutput(self.embed(input_ids.to(self.embed.weight.device)))


class GreyCodecVAE(nn.Module):
    """Stand-in VAE: latent = greyscale image downscaled 8x, broadcast over the 16 latent channels."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.zeros(1))

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def encode_pixels_to_latents(self, pixels):
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        pixels = pixels.to(self.device, dtype=self._dummy.dtype)
        grey = pixels.mean(dim=1, keepdim=True)  # (B, 1, H, W) in [-1, 1]
        latent = F.avg_pool2d(grey, 8)
        return latent.repeat(1, 16, 1, 1)

    def decode_to_pixels(self, latents):
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        grey = latents.to(self.device, dtype=self._dummy.dtype).mean(dim=1, keepdim=True)
        pixels = F.interpolate(grey, scale_factor=8, mode="nearest")
        return pixels.repeat(1, 3, 1, 1)


def build_small_anima():
    return anima_models.Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=8,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=256,
        num_blocks=6,
        num_heads=8,
        crossattn_emb_channels=QWEN3_HIDDEN,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        adaln_lora_dim=64,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        attn_mode="torch",
    )


def make_mnist_dataset(root: str, num_images: int, size: int):
    """1k MNIST digits as RGB PNGs with captions, in the DreamBooth layout."""
    image_dir = os.path.join(root, "1_digits")
    os.makedirs(image_dir, exist_ok=True)

    rows = np.loadtxt(MNIST_CSV, delimiter=",", max_rows=num_images, dtype=np.int32)
    labels = rows[:, 0]
    pixels = rows[:, 1:].reshape(-1, 28, 28).astype(np.uint8)

    paths_by_label = {}
    for i, (label, digit) in enumerate(zip(labels, pixels)):
        image = Image.fromarray(digit).convert("RGB").resize((size, size), Image.NEAREST)
        path = os.path.join(image_dir, f"{i:05d}_{label}.png")
        image.save(path)
        with open(os.path.splitext(path)[0] + ".txt", "w") as f:
            f.write("a handwritten digit")  # deliberately label-free: only DINOv3 knows which digit it is
        paths_by_label.setdefault(int(label), []).append(path)

    print(f"dataset: {len(labels)} images at {size}x{size}, digits {sorted(paths_by_label)}")
    return image_dir, paths_by_label


captured = {}


def install_stubs():
    from library import anima_utils as au

    tokenizer = au.load_qwen3_tokenizer(os.path.join(REPO_ROOT, "configs", "qwen3_06b"))

    def fake_load_qwen3_text_encoder(qwen3_path, dtype=torch.bfloat16, device="cpu", **kwargs):
        torch.manual_seed(0)
        return StubQwen3().to(device=device, dtype=dtype), tokenizer

    def fake_load_anima_model(device, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, **kwargs):
        torch.manual_seed(0)
        return build_small_anima()

    def fake_load_vae(args, device="cpu", disable_mmap=True):
        return GreyCodecVAE().to(device)

    train_anima_dinov3.anima_utils.load_qwen3_text_encoder = fake_load_qwen3_text_encoder
    train_anima_dinov3.anima_utils.load_anima_model = fake_load_anima_model
    train_anima_dinov3.anima_train_utils.load_qwen_image_vae = fake_load_vae
    anima_train_utils.anima_utils.save_anima_model = lambda ckpt_file, sd, metadata, dtype: torch.save(sd, ckpt_file)

    # capture the trained model and the loss curve for the post-training checks
    original_create = anima_dinov3.create_dinov3_model

    def capturing_create(dit, encoder, args):
        model = original_create(dit, encoder, args)
        captured["model"] = model
        captured["encoder"] = encoder
        return model

    train_anima_dinov3.anima_dinov3.create_dinov3_model = capturing_create

    losses = []
    original_add = logging_util.LossRecorder.add

    def recording_add(self, *, epoch, step, loss):
        losses.append(loss)
        return original_add(self, epoch=epoch, step=step, loss=loss)

    logging_util.LossRecorder.add = recording_add
    captured["losses"] = losses


@torch.no_grad()
def condition_ablation(model, dataset_dir, image_paths, device, num_batches=12, batch_size=8):
    """Does the model actually use the DINOv3 condition? Same noise, same timesteps, shuffled embeddings."""
    from library import strategy_base

    te_strategy = strategy_base.TextEncoderOutputsCachingStrategy.get_strategy()
    dino_strategy = te_strategy.dinov3_strategy
    vae = GreyCodecVAE().to(device)

    model.eval()
    matched_losses, shuffled_losses = [], []

    for b in range(num_batches):
        batch_paths = image_paths[b * batch_size : (b + 1) * batch_size]
        if len(batch_paths) < batch_size:
            break

        images = []
        dino = []
        for path in batch_paths:
            image = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            dino.append(torch.from_numpy(dino_strategy.load_embeds_npz(dino_strategy.get_embeds_npz_path(path))))
        latents = vae.encode_pixels_to_latents(torch.stack(images).to(device))
        dino = torch.stack(dino).to(device, dtype=torch.float32)

        te_npz = te_strategy.load_outputs_npz(os.path.splitext(batch_paths[0])[0] + te_strategy.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX)
        prompt_embeds = torch.from_numpy(te_npz[0]).unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # same caption for all

        generator = torch.Generator(device=device).manual_seed(1234 + b)
        noise = torch.randn(latents.shape, generator=generator, device=device)
        sigmas = torch.rand(batch_size, generator=generator, device=device).view(-1, 1, 1, 1)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents
        timesteps = sigmas.view(-1)
        padding_mask = torch.zeros(batch_size, 1, latents.shape[-2], latents.shape[-1], device=device)

        for name, embeds, sink in [
            ("matched", dino, matched_losses),
            ("shuffled", dino.roll(1, dims=0), shuffled_losses),  # each image gets another image's embedding
        ]:
            pred = model(
                noisy.unsqueeze(2),
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                dinov3_embeds=embeds,
            ).squeeze(2)
            sink.append(F.mse_loss(pred.float(), target.float()).item())

    return float(np.mean(matched_losses)), float(np.mean(shuffled_losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="/content/mnist_dinov3_out")
    parser.add_argument("--data_dir", type=str, default="/content/mnist_dinov3_data")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])
    args = parser.parse_args()

    assert torch.cuda.is_available(), "this test needs a GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    install_stubs()
    image_dir, paths_by_label = make_mnist_dataset(args.data_dir, args.num_images, args.resolution)
    os.makedirs(args.output_dir, exist_ok=True)

    # sample prompts: the same caption, but conditioned on reference images of different digits.
    # If the DINOv3 branch works, the samples should follow the reference digit.
    prompts_path = os.path.join(args.data_dir, "prompts.txt")
    with open(prompts_path, "w") as f:
        for digit in [0, 1, 7]:
            reference = paths_by_label[digit][0]
            f.write(f"a handwritten digit --w {args.resolution} --h {args.resolution} --s 20 --l 1.0 --d 7 --ci {reference}\n")
        f.write(f"a handwritten digit --w {args.resolution} --h {args.resolution} --s 20 --l 1.0 --d 7\n")  # null condition

    argv = [
        "--pretrained_model_name_or_path", "stub_dit.safetensors",
        "--qwen3", os.path.join(REPO_ROOT, "configs", "qwen3_06b"),
        "--vae", "stub_vae.safetensors",
        "--train_data_dir", args.data_dir,
        "--resolution", f"{args.resolution},{args.resolution}",
        "--output_dir", args.output_dir,
        "--output_name", "mnist_dinov3",
        "--train_batch_size", str(args.batch_size),
        "--max_train_steps", str(args.max_train_steps),
        "--learning_rate", "1e-4",
        "--dinov3_projector_lr", "1e-4",
        "--optimizer_type", "adamw",
        "--mixed_precision", args.mixed_precision,  # T4: no bf16
        "--save_precision", "float",
        "--cache_latents", "--cache_latents_to_disk",
        "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
        "--vae_batch_size", "16",  # the default of 1 makes latent caching CPU-bound and slow
        "--text_encoder_batch_size", "16",
        "--dinov3_token_mode", "cls",
        "--dinov3_cond_mode", "concat",
        "--dinov3_image_size", "224",
        "--dinov3_batch_size", "16",
        "--dinov3_dropout_rate", "0.1",
        "--qwen3_max_token_length", "64",
        "--t5_max_token_length", "64",
        "--sample_prompts", prompts_path,
        "--sample_every_n_steps", str(max(1, args.max_train_steps // 2)),
        "--max_data_loader_n_workers", "2",
        "--seed", "42",
    ]

    start = time.time()
    parser = train_anima_dinov3.setup_parser()
    train_args = parser.parse_args(argv)
    train_anima_dinov3.train(train_args)
    minutes = (time.time() - start) / 60
    print(f"\ntraining wall clock: {minutes:.1f} min")

    losses = captured["losses"]
    chunk = max(1, len(losses) // 5)
    curve = [float(np.mean(losses[i : i + chunk])) for i in range(0, len(losses) - chunk + 1, chunk)]
    print(f"loss over training (mean per {chunk} steps): " + " -> ".join(f"{v:.4f}" for v in curve))

    model = captured["model"]
    device = torch.device("cuda")
    model.to(device, dtype=torch.float32)
    all_paths = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")
    )
    matched, shuffled = condition_ablation(model, image_dir, all_paths, device)
    print(f"\nDINOv3 condition ablation (lower is better):")
    print(f"  loss with matching embeddings : {matched:.4f}")
    print(f"  loss with shuffled embeddings : {shuffled:.4f}")
    print(f"  relative degradation          : {(shuffled - matched) / matched * 100:.1f}%")
    if shuffled > matched * 1.02:
        print("  -> the model relies on the DINOv3 condition")
    else:
        print("  -> WARNING: the DINOv3 condition makes little difference")

    print(f"\nsamples: {sorted(os.listdir(os.path.join(args.output_dir, 'sample')))}")


if __name__ == "__main__":
    main()
