"""The video VAE's checkpoint-to-diffusers key plan.

The same shape of problem as the transformer, and it needs the same three careful transforms, so it
reuses ``models/keymap.py`` rather than growing its own loader:

* the encoder's CNN levels are spelled the CompVis way (``down.{i}.block.{j}``) and become the
  diffusers idiom (``down_blocks.{i}.resnets.{j}``),
* the ViT decoder's fused ``attn.to_qkv`` is **per-head interleaved**, exactly like the DiT's, and
  splits into ``to_q``/``to_k``/``to_v``,
* the gated FFN's ``ff.w1`` halves swap for diffusers' ``SwiGLU``.

Mapped from upstream's own ``convert_minimax_h3_to_diffusers.py`` rather than inferred, because the
published file is written for MiniMax's implementation and a wrong split here decodes to a plausible
but wrong picture.
"""

from __future__ import annotations

import re

from ..keymap import (
    Drop,
    KeyPlan,
    Rename,
    RowLayout,
    Split,
    SplitWeightNorm,
    SwapHalves,
)

PLAN_VERSION = "minimax-h3.vae-keys.1"

#: From the file's own embedded `source_config.vit_decoder_kwargs`: 32 heads of 64.
DECODER_HEADS = 32
DECODER_HEAD_DIM = 64

#: Present in the file but not loaded as tensors. The latent statistics are constructor arguments on
#: the port, read from the same embedded metadata, so loading them again would be a second copy.
DROPPED = {
    "decoder.mask_token": "the port allocates its own mask token",
    "latents_mean": "read from the file's metadata as a config value, not a parameter",
    "latents_std": "read from the file's metadata as a config value, not a parameter",
}


def rename(source_key: str) -> str:
    """One original key onto its diffusers module path. No tensor transform."""
    target = source_key
    if target.startswith("encoder.down."):
        level, rest = target.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target = f"encoder.down_blocks.{level}.{rest}"
    target = target.replace("decoder.x_embedder.", "decoder.proj_in.")
    target = target.replace(".attn.to_out.", ".attn.to_out.0.")
    target = target.replace(".ff.w1.", ".ff.net.0.proj.")
    target = target.replace(".ff.w2.", ".ff.net.2.")
    return target


def build_plan(source_keys: list[str]) -> KeyPlan:
    """The plan for whatever this file actually contains.

    Driven off the file's own key list rather than a hardcoded layer count, because the VAE's depth
    is not stated anywhere we read and a future build may differ.
    """
    actions: dict[str, object] = {}
    for key in source_keys:
        if key in DROPPED:
            actions[key] = Drop(DROPPED[key])
            continue
        if ".attn.to_qkv." in key:
            prefix, suffix = key.split(".attn.to_qkv.")
            actions[key] = Split(
                (
                    f"{prefix}.attn.to_q.{suffix}",
                    f"{prefix}.attn.to_k.{suffix}",
                    f"{prefix}.attn.to_v.{suffix}",
                ),
                layout=RowLayout.INTERLEAVED,
                head_dim=DECODER_HEAD_DIM,
            )
            continue
        target = rename(key)
        # `w1` fuses gate and up; diffusers' SwiGLU reads them the other way round.
        actions[key] = SwapHalves(target) if ".ff.w1." in key else Rename(target)
    return KeyPlan(version=PLAN_VERSION, actions=actions)  # type: ignore[arg-type]


def build_audio_plan(source_keys: list[str], target_keys: list[str]) -> KeyPlan:
    """The audio VAE's plan: pass-through, except that the port weight-normalises its convolutions.

    Upstream's converter only *validates* the audio VAE keys against a freshly built model rather
    than renaming them, which is true of the original checkpoint. The consolidated repack ships the
    **fused** weights instead, so every weight-normalised convolution has to be split back into the
    magnitude and direction the port's parameters expect. There are 172 of them.
    """
    weight_normed = {
        key[: -len("_g")] for key in target_keys if key.endswith(".weight_g")
    }
    actions: dict[str, object] = {}
    for key in source_keys:
        if key in AUDIO_DROPPED:
            actions[key] = Drop(AUDIO_DROPPED[key])
        elif key in weight_normed:
            actions[key] = SplitWeightNorm(f"{key}_g", f"{key}_v")
        else:
            actions[key] = Rename(key)
    return KeyPlan(version=f"{PLAN_VERSION}.audio", actions=actions)  # type: ignore[arg-type]


#: Same reasoning as the video VAE's: statistics the port takes as configuration.
AUDIO_DROPPED = {
    "latents_mean": "read from the file's metadata as a config value, not a parameter",
    "latents_std": "read from the file's metadata as a config value, not a parameter",
}


def self_computed_targets(target_keys: list[str]) -> set[str]:
    """Buffers the port derives itself, which the plan is not expected to fill.

    The rotary tables in the ViT decoder are a pure function of the geometry, the same way the DiT's
    ``rope.inv_freq`` is.
    """
    return {key for key in target_keys if re.search(r"(^|\.)(rope|freqs|inv_freq)", key)}
