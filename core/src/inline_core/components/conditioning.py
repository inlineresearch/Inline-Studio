"""Opaque conditioning and latents. Conditioning is family-defined; the graph never inspects it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Conditioning:
    """Opaque output of a TextEncoder, consumed by a Denoiser. No fixed tensor shape."""


@dataclass
class Latents:
    """A latent tensor. Its device and dtype live on the tensor, set by the policy at load."""

    tensor: torch.Tensor
