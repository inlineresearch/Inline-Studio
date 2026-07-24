"""What differs between the architectures we train LoRAs for.

Z-Image and Krea 2 share the whole harness - dataset export, precache-then-free, PEFT adapter,
checkpoint/resume, the JSON-line protocol - and differ in only four things: which Linears to adapt,
how a noise level maps to a timestep, what the model is asked to predict, and how one forward call
is shaped. Those four live here so ``trainer.py`` stays one loop.

Both are rectified flow, but with **opposite conventions**, which is exactly the kind of detail a
test should pin: Z-Image predicts ``clean - noise`` at timestep ``1 - sigma``, Krea 2 predicts
``noise - clean`` at timestep ``sigma``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Z_IMAGE = "z-image"
KREA2 = "krea2"

#: Z-Image: every ZImageTransformerBlock's attention + SwiGLU feed-forward Linears, confirmed
#: against ZImageTransformer2DModel.named_modules() (34 blocks, 238 Linears).
_ZIMAGE_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]

#: Krea 2: the authors' recommended default, as shipped in diffusers'
#: examples/dreambooth/train_dreambooth_lora_krea2.py. Adapting the text-fusion stage and the
#: img/txt/time projections (not just attention) is what the official style LoRAs do too.
_KREA2_TARGETS = [
    "img_in",
    "final_layer.linear",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_gate",
    "ff.up",
    "ff.down",
    "text_fusion.projector",
    "txt_in.linear_1",
    "txt_in.linear_2",
    "time_embed.linear_1",
    "time_embed.linear_2",
    "time_mod_proj",
]


#: The attention projections shared by both architectures. Narrowing to these is the Krea 2
#: authors' advice for long runs: adapting the feed-forward and projection layers too is stronger
#: on short style runs but costs prompt adherence as the run goes on.
_ATTENTION = ("to_q", "to_k", "to_v", "to_out.0", "to_gate")


@dataclass(frozen=True)
class TrainingArch:
    """One architecture's training behaviour."""

    key: str
    target_modules: list[str]
    #: (device, shift) -> a scalar noise fraction in (0, 1).
    sigma: Callable[[Any, float], Any]
    #: sigma -> the normalized timestep the model expects alongside that noise level.
    timestep: Callable[[Any], Any]
    #: (clean, noise) -> what the model is trained to predict.
    target: Callable[[Any, Any], Any]
    #: (transformer, noisy, timestep, item) -> the prediction, same shape as the clean latent.
    forward: Callable[..., Any]


# --- Z-Image -------------------------------------------------------------------------------------


def _zimage_sigma(device: Any, shift: float) -> Any:
    import torch

    # Logit-normal noise fraction (denser near the middle, as flow-match trainers favor), then
    # Z-Image's static resolution shift so training visits the noise levels inference does.
    u = torch.sigmoid(torch.randn((), device=device))
    return shift * u / (1.0 + (shift - 1.0) * u)


def _zimage_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from ZImageTransformer2DModel, mirroring ZImagePipeline's call.

    The model takes per-image LISTS - latents as (C, F, H, W) with a temporal axis, captions as
    (seq, dim) - and returns per-image latents in ``.sample``."""
    out = transformer(
        [noisy.unsqueeze(1)],  # (C, H, W) -> [(C, 1, H, W)]
        timestep.reshape(1),
        [item["embed"]],
        return_dict=True,
    )
    sample = out.sample if hasattr(out, "sample") else out[0]
    return sample[0].squeeze(1)


# --- Krea 2 --------------------------------------------------------------------------------------

_PATCH = 2


def _krea2_sigma(device: Any, shift: float) -> Any:
    import torch

    # The diffusers reference samples an index into the *unshifted* 1000-step schedule with a
    # logit-normal density, which is this sigma directly (at init `timesteps == sigmas * 1000` when
    # use_dynamic_shifting is on). `shift` is unused - Krea 2's shift is resolution dependent and
    # applied at inference, not baked into the training distribution.
    del shift
    return torch.sigmoid(torch.randn((), device=device))


def _krea2_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from Krea2Transformer2DModel: pack the latent into 2x2 patches, run the
    combined [text | image] sequence, unpack back to a latent grid."""
    from diffusers import Krea2Pipeline

    channels, height, width = noisy.shape
    grid_h, grid_w = height // _PATCH, width // _PATCH
    packed = (
        noisy.view(1, channels, grid_h, _PATCH, grid_w, _PATCH)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, grid_h * grid_w, channels * _PATCH * _PATCH)
    )
    embed, mask = item["embed"], item["mask"]
    position_ids = Krea2Pipeline.prepare_position_ids(
        embed.shape[0], grid_h, grid_w, noisy.device
    )
    out = transformer(
        hidden_states=packed,
        encoder_hidden_states=embed.unsqueeze(0),
        timestep=timestep.reshape(1),
        position_ids=position_ids,
        encoder_attention_mask=mask.unsqueeze(0),
        return_dict=False,
    )[0]
    return (
        out.view(1, grid_h, grid_w, channels, _PATCH, _PATCH)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(channels, height, width)
    )


ARCHS: dict[str, TrainingArch] = {
    Z_IMAGE: TrainingArch(
        key=Z_IMAGE,
        target_modules=_ZIMAGE_TARGETS,
        sigma=_zimage_sigma,
        # 1 = clean, 0 = noise; the model applies t_scale itself, so this must not be pre-scaled.
        timestep=lambda sigma: 1.0 - sigma,
        # The pipeline negates the output before handing it to FlowMatchEuler as the velocity.
        target=lambda clean, noise: clean - noise,
        forward=_zimage_forward,
    ),
    KREA2: TrainingArch(
        key=KREA2,
        target_modules=_KREA2_TARGETS,
        sigma=_krea2_sigma,
        timestep=lambda sigma: sigma,
        target=lambda clean, noise: noise - clean,
        forward=_krea2_forward,
    ),
}


def get(key: str | None) -> TrainingArch:
    """The arch to train. Defaults to Z-Image so a run predating Krea 2 still resumes."""
    arch = ARCHS.get(key or Z_IMAGE)
    if arch is None:
        raise RuntimeError(f"Unknown training architecture {key!r}.")
    return arch


def target_modules(arch: TrainingArch, scope: str | None) -> list[str]:
    """The Linears the adapter attaches to. ``attention`` keeps only the attention projections the
    arch actually has, so the same scope means the same thing across architectures."""
    if (scope or "full") == "full":
        return arch.target_modules
    if scope != "attention":
        raise RuntimeError(f"Unknown LoRA scope {scope!r}.")
    narrowed = [m for m in arch.target_modules if m in _ATTENTION]
    if not narrowed:  # defensive: an arch with no matching attention names would train nothing
        raise RuntimeError(f"{arch.key} has no attention modules to narrow to.")
    return narrowed
