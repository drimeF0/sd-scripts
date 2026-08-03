from contextlib import nullcontext
from types import SimpleNamespace

import torch

from library.xm import (
    compute_anima_xm_loss,
    compute_anima_xm_loss_for_network,
    validate_xm_args,
    xm_chunked_best_of_k,
)


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def print(*args, **kwargs):
        pass


class _Scheduler:
    config = SimpleNamespace(num_train_timesteps=1000)


class _AnimaStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, timesteps, prompt_embeds, **kwargs):
        return x * self.scale


def _args(**overrides):
    values = dict(
        xm_best_of_k=3,
        xm_chunk_bs_mult=2,
        xm_save_mem_mode=True,
        xm_debug_mode=True,
        gradient_checkpointing=False,
        weighting_scheme="none",
        loss_type="l2",
        masked_loss=False,
        timestep_sampling="uniform",
        timestep_total_blocks=None,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        huber_schedule="constant",
        huber_c=0.1,
        huber_scale=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _conditions(batch_size):
    return [
        torch.randn(batch_size, 3, 4),
        torch.ones(batch_size, 3),
        torch.ones(batch_size, 3, dtype=torch.long),
        torch.ones(batch_size, 3),
    ]


def test_chunked_best_of_k_selects_each_samples_minimum_and_structured_prediction():
    torch.manual_seed(123)
    batch_size, best_of_k = 2, 3
    explored_losses = []

    def calculate(conditions, samples, learning, rand_inputs, rand_seeds):
        losses = rand_inputs.square().flatten(1).mean(1)
        explored_losses.append(losses)
        return losses, (rand_inputs, {"seeds": rand_seeds})

    losses, predictions = xm_chunked_best_of_k(
        calculate,
        (torch.zeros(batch_size, 1), None),
        torch.zeros(batch_size, 1, 2, 2),
        best_of_k,
        best_of_k,
        save_mem_mode=False,
    )

    all_losses = torch.cat(explored_losses).reshape(best_of_k, batch_size)
    assert torch.equal(losses, all_losses.min(dim=0).values)
    assert predictions[0].shape == (batch_size, 1, 2, 2)
    assert predictions[1]["seeds"].shape == (batch_size,)


def test_xm_argument_validation_rejects_a_chunked_differentiable_graph():
    args = _args(xm_save_mem_mode=False, xm_chunk_bs_mult=2, xm_best_of_k=3, xm_debug_mode=False)
    try:
        validate_xm_args(args)
    except ValueError as error:
        assert "fit in one chunk" in str(error)
    else:
        raise AssertionError("invalid XM arguments were accepted")


def test_anima_network_xm_preserves_training_graph_with_and_without_recompute():
    torch.manual_seed(456)
    batch_size = 2
    latents = torch.randn(batch_size, 1, 2, 2)

    for save_mem_mode in (False, True):
        model = _AnimaStub()
        prediction, target, _, weighting = compute_anima_xm_loss_for_network(
            _args(
                xm_save_mem_mode=save_mem_mode,
                xm_debug_mode=save_mem_mode,
                xm_chunk_bs_mult=3,
                ip_noise_gamma=0.1,
            ),
            _Accelerator(),
            _Scheduler(),
            latents,
            {"loss_weights": torch.ones(batch_size)},
            _conditions(batch_size),
            model,
            torch.float32,
            train_unet=True,
        )

        assert prediction.requires_grad
        assert prediction.shape == target.shape == latents.shape
        ((prediction - target).square() * weighting).mean().backward()
        assert model.scale.grad is not None
        assert torch.isfinite(model.scale.grad)


def test_anima_full_xm_supports_ip_noise_and_mask_across_multiple_chunks():
    torch.manual_seed(789)
    batch_size = 2
    latents = torch.randn(batch_size, 1, 2, 2)
    model = _AnimaStub()
    args = _args(ip_noise_gamma=0.1, masked_loss=True, loss_type="huber")
    batch = {
        "loss_weights": torch.tensor([1.0, 0.5]),
        "alpha_masks": torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]),
    }
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = _conditions(batch_size)

    loss = compute_anima_xm_loss(
        args,
        model,
        latents,
        _Scheduler(),
        prompt_embeds,
        attn_mask,
        t5_input_ids,
        t5_attn_mask,
        torch.zeros(batch_size, 1, 2, 2),
        _Accelerator(),
        torch.float32,
        batch=batch,
    )

    assert loss.ndim == 0
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
