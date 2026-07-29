"""Krea 2 depth control: the public ``Patil/Krea-2-depth-controlnet`` control-LoRA, ported onto the
diffusers ``Krea2Transformer2DModel``.

The adapter is a rank-64 LoRA on every transformer block **plus a full replacement input
projection** (``first.weight [6144, 128]``): the base projection takes 64 packed latent channels,
expanded to 128 so a VAE-encoded depth latent rides alongside the noisy latent, concatenated on the
channel dim. The base stays frozen and the depth latent is constant across the whole denoise.

The checkpoint uses the reference krea-2 names (``blocks.N.attn.wq``, ``first``); ``convert_key``
already remaps those to diffusers names (``transformer_blocks.N.attn.to_q``, ``img_in``), the same
rename the base checkpoint needs. Wrapping ``img_in`` lets the depth concat happen transparently
inside the pipeline's own denoise loop - no fork of ``Krea2Pipeline.__call__``. Reference:
github.com/Tanmaypatil123/Krea-2-controlnet.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...errors import ComponentError
from .convert import convert_key
from .img2img import encode_image


class LoRALinear(nn.Module):
    """``y = W x + scale * (x A^T) B^T``. Base frozen; ``scale`` is the live control strength."""

    def __init__(self, base: nn.Linear, rank: int, scale: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.scale = scale
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)
        return self.base(x) + lora * self.scale


class ControlInputLayer(nn.Module):
    """Replaces ``img_in``: input width doubled (64 -> 128) to accept
    ``[noisy latent ; depth latent]`` concatenated on the channel dim. The depth latent is stashed
    on ``self.ctrl`` once per run and broadcast over the batch (so CFG's doubled batch lines up)."""

    def __init__(self, pretrained: nn.Linear) -> None:
        super().__init__()
        self._in = pretrained.in_features
        # The checkpoint always carries the full trained input projection (both the base and depth
        # halves), so zeros are enough here - and this avoids dequantizing a possibly-int8 base.
        self.weight = nn.Parameter(torch.zeros(pretrained.out_features, self._in * 2))
        self.bias = nn.Parameter(torch.zeros(pretrained.out_features))
        self.ctrl: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ctrl is None:  # a control pipe always has ctrl set before a run; zeros = no depth
            ctrl = x.new_zeros((*x.shape[:-1], self._in))
        else:
            ctrl = self.ctrl.to(device=x.device, dtype=x.dtype)
            if ctrl.shape[0] != x.shape[0]:
                ctrl = ctrl.expand(x.shape[0], -1, -1)
        combined = torch.cat([x, ctrl], dim=-1)
        return F.linear(combined, self.weight.to(x.dtype), self.bias.to(x.dtype))


def _get(root: Any, path: str) -> Any:
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def _set(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = _get(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def _module_paths(state: dict[str, torch.Tensor]) -> list[str]:
    """The block-linear module paths the adapter references (``img_in`` handled separately)."""
    paths: list[str] = []
    for key in state:
        if key.startswith("img_in."):
            continue
        stem = key.rsplit(".", 1)[0]  # drop the trailing .A / .B
        if stem not in paths:
            paths.append(stem)
    return paths


def install_depth_control(transformer: Any, lora_file: str) -> None:
    """Wrap ``img_in`` + every block linear the LoRA touches, then load the trained weights in.

    Idempotent per pipeline: only called on a cache miss, so a cached control pipe keeps its work.
    ``set_control`` / ``set_control_strength`` are the per-run knobs on top of it.
    """
    import safetensors.torch

    st: Any = safetensors.torch
    raw: dict[str, Any] = st.load_file(lora_file)
    state = {convert_key(k): v for k, v in raw.items()}
    a_shapes = [t.shape[0] for k, t in state.items() if k.endswith(".A")]
    if not a_shapes:
        raise ComponentError("Krea 2 depth control file has no LoRA tensors; not the adapter.")
    rank = a_shapes[0]

    device = next(transformer.parameters()).device
    transformer.img_in = ControlInputLayer(transformer.img_in).to(device)
    for path in _module_paths(state):
        _set(transformer, path, LoRALinear(_get(transformer, path), rank).to(device))

    missing, unexpected = transformer.load_state_dict(state, strict=False)
    if unexpected:
        raise ComponentError(
            f"Krea 2 depth control-LoRA has {len(unexpected)} tensors that do not map onto the "
            f"model (e.g. {', '.join(unexpected[:3])}). It may be for a different Krea 2 build."
        )
    # The expanded input projection is zero-initialised, so it MUST come from the checkpoint.
    if "img_in.weight" in missing or "img_in.bias" in missing:
        raise ComponentError(
            "Krea 2 depth control-LoRA is missing its input projection (img_in); not the adapter."
        )


def set_control_strength(transformer: Any, scale: float) -> None:
    """Dial the block LoRA delta (the depth ``img_in`` expansion always applies). No rebuild."""
    for module in transformer.modules():
        if isinstance(module, LoRALinear):
            module.scale = scale


def set_control(transformer: Any, ctrl_latent: torch.Tensor | None) -> None:
    img_in = transformer.img_in
    if isinstance(img_in, ControlInputLayer):
        img_in.ctrl = ctrl_latent


def encode_depth_latent(
    pipe: Any, depth_image: Any, *, width: int, height: int, device: str, generator: Any
) -> torch.Tensor:
    """The depth map as a packed Krea 2 latent, ready to concat onto the noisy latent - the same VAE
    encode + pack the img2img path uses, so the two latents share a layout."""
    return encode_image(pipe, depth_image, width, height, device, generator)
