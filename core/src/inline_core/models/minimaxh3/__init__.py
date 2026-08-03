"""MiniMax H3: four local video nodes, backed by code vendored from an unmerged diffusers PR."""

from __future__ import annotations

from .runner import DESCRIPTORS, VARIANTS, register_minimax_h3

__all__ = ["DESCRIPTORS", "VARIANTS", "register_minimax_h3"]
