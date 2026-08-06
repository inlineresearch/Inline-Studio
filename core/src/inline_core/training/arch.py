"""What differs between the architectures we train LoRAs for.

Every arch shares the whole harness - dataset export, precache-then-free, PEFT adapter,
checkpoint/resume, the JSON-line protocol - and they differ in only four things: which Linears to
adapt, how a noise level maps to a timestep, what the model is asked to predict, and how one forward
call is shaped. Those four live here so ``trainer.py`` stays one loop.

All four are rectified flow, but with **opposite conventions**, which is exactly the kind of detail
a test should pin: Z-Image and MiniMax H3 predict ``clean - noise`` at timestep ``1 - sigma``, while
Krea 2 and FLUX.2 predict ``noise - clean`` at timestep ``sigma``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Z_IMAGE = "z-image"
KREA2 = "krea2"
FLUX2 = "flux2"
MINIMAX_H3 = "minimax-h3"

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


#: FLUX.2: every Linear in the MMDiT stack, confirmed against Flux2Transformer2DModel's own
#: named_parameters (5 double + 20 single blocks on klein 4B, 169 tensors, no biases anywhere).
#: Double blocks carry separate image/text streams (``add_*`` is the text side); single blocks fuse
#: QKV and the MLP into one ``to_qkv_mlp_proj``, which is why they cannot be adapted attention-only.
#:
#: The single blocks' output projection is deliberately absent. PEFT matches targets by module-name
#: suffix, and a single block's is ``attn.to_out`` (a Linear) while a double block's is also
#: ``attn.to_out`` (a ModuleList of Linear + Dropout, which PEFT refuses) - no suffix separates
#: them. The single blocks still learn through ``to_qkv_mlp_proj``, their dominant projection.
_FLUX2_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff.linear_in",
    "ff.linear_out",
    "ff_context.linear_in",
    "ff_context.linear_out",
    "to_qkv_mlp_proj",
    "x_embedder",
    "context_embedder",
    "proj_out",
]


#: MiniMax H3: every Linear in a block except ``adaln_proj``, plus the text projection.
#:
#: ``adaln_proj`` is out because the load factorises it to rank 8, so a LoRA would attach to eight
#: columns carrying the whole modulation signal. The fp32-pinned heads are out because adapting them
#: fights the checkpoint's precision split. PEFT matches by suffix, so this also reaches the two
#: token-refiner blocks, which is why the loader has to fuse outside the block stack.
_MINIMAX_H3_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "context_embedder",
]


#: The attention projections, across architectures. Narrowing to these is the Krea 2 authors' advice
#: for long runs: adapting the feed-forward and projection layers too is stronger on short style
#: runs but costs prompt adherence as the run goes on.
#:
#: ``to_qkv_mlp_proj`` is deliberately absent: FLUX.2's single blocks fuse attention and MLP into
#: that one projection, so including it would silently make "attention" mean "everything". On
#: FLUX.2 this scope therefore adapts the double blocks only - the text/image fusion stage.
_ATTENTION = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_gate",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


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



# --- FLUX.2 ---------------------------------------------------------------------------------------


def _flux2_sigma(device: Any, shift: float) -> Any:
    import torch

    # Logit-normal, matching the reference trainers. FLUX.2's shift is resolution dependent and
    # computed at inference (``compute_empirical_mu``), so it is not baked into the training
    # distribution - the same reasoning as Krea 2.
    del shift
    return torch.sigmoid(torch.randn((), device=device))


def _flux2_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from Flux2Transformer2DModel, mirroring Flux2KleinPipeline's denoise call.

    The latent arrives already patchified and batch-norm normalized by the precache (see
    ``dataset._flux2_latent``), so it is (128, H/16, W/16) and packing is the pipeline's plain
    flatten to (1, tokens, 128). ``timestep`` is the raw sigma: the pipeline passes
    ``timestep / 1000`` where its own timesteps are ``sigma * 1000``.

    The packing and position-id helpers come from the pipeline itself rather than a local copy, so
    training cannot drift from inference.
    """
    from diffusers import Flux2KleinPipeline as P

    latents = noisy.unsqueeze(0)  # (C, h, w) -> (1, C, h, w)
    packed = P._pack_latents(latents)
    img_ids = P._prepare_latent_ids(latents).to(packed.device)
    embed = item["embed"]
    embed = embed.unsqueeze(0) if embed.dim() == 2 else embed
    txt_ids = P._prepare_text_ids(embed).to(packed.device)

    out = transformer(
        hidden_states=packed,
        encoder_hidden_states=embed,
        timestep=timestep.reshape(1),
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=None,
        return_dict=False,
    )[0]
    channels, height, width = noisy.shape
    # Unpack: (1, tokens, C) -> (C, h, w), the inverse of the flatten above.
    return out[:, : height * width].permute(0, 2, 1).reshape(channels, height, width)



# --- MiniMax H3 -----------------------------------------------------------------------------------

#: H3's (t, h, w) patch. A still is one latent frame, so only the spatial half ever bites.
_H3_PATCH = (1, 2, 2)


def _h3_forward(transformer: Any, noisy: Any, timestep: Any, item: dict[str, Any]) -> Any:
    """One prediction from MiniMaxH3Transformer3DModel, mirroring the vendored denoise block.

    H3 packs text, audio and video into one 1-D sequence and the caller owns that layout, so the
    precache builds it per image and a step only patchifies and selects the video rows back out.
    """
    from ..models.minimaxh3.vendor.packing import patchify_video_latents, unpatchify_video_tokens

    channels, frames, height, width = noisy.shape
    rows = patchify_video_latents(noisy.unsqueeze(0), _H3_PATCH)

    # One distinct noise level, so every row indexes timestep 0. Training pins no conditioning rows
    # and a still has no audio rows, which is what collapses the vendored (timestep, index) plan to
    # this; ``timestep_indices`` comes from the precache, which derives it from build_row_timesteps
    # rather than assuming the collapse.
    video, _audio = transformer(
        hidden_states=rows.unsqueeze(0),
        audio_hidden_states=item["audio"].unsqueeze(0),
        encoder_hidden_states=item["embed"].unsqueeze(0),
        timestep=timestep.reshape(1),
        timestep_indices=item["timestep_indices"],
        token_tags=item["token_tags"],
        position_ids=item["position_ids"],
        video_indices=item["video_indices"],
        audio_indices=item["audio_indices"],
        text_indices=item["text_indices"],
        return_dict=False,
    )
    unpacked = unpatchify_video_tokens(video[0], frames, height, width, channels, _H3_PATCH)
    return unpacked[0]


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
    FLUX2: TrainingArch(
        key=FLUX2,
        target_modules=_FLUX2_TARGETS,
        sigma=_flux2_sigma,
        # Rectified flow with Krea 2's convention: x_t = (1 - sigma) * clean + sigma * noise, so
        # d x_t / d sigma is noise - clean, and the model is called at the sigma itself.
        timestep=lambda sigma: sigma,
        target=lambda clean, noise: noise - clean,
        forward=_flux2_forward,
    ),
    MINIMAX_H3: TrainingArch(
        key=MINIMAX_H3,
        target_modules=_MINIMAX_H3_TARGETS,
        # Same shift expression as Z-Image, at the scheduler's video shift of 12.0.
        sigma=_zimage_sigma,
        # Z-Image's convention, opposite to Krea 2 and FLUX.2, and pinned against the vendored
        # scheduler in test_minimaxh3_training.py rather than restated here.
        timestep=lambda sigma: 1.0 - sigma,
        target=lambda clean, noise: clean - noise,
        forward=_h3_forward,
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
