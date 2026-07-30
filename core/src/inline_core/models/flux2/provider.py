"""FLUX.2's answer to "what do I need on disk" - the model popup's data source for the node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent
from .requirements import (
    flux2_requirements,
    footprint_bytes,
    resolve_diffusion,
    resolve_text_encoder,
    resolve_vae,
)


class Flux2Provider:
    """Requirements + fit estimate for the FLUX.2 node.

    One provider covers every variant: the popup shows the required set for whichever checkpoint is
    installed, plus the rest of the family as optional downloads.
    """

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return flux2_requirements(params)

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether the installed checkpoint fits this machine, and how, so the popup warns before a
        load. Pure ``stat`` plus a live VRAM/RAM probe; None when it cannot be sized."""
        if policy is None:
            return None
        try:
            from ...device.policy import ModelFootprint
        except ImportError:
            return None
        footprint = ModelFootprint(
            **footprint_bytes(
                resolve_diffusion(None), resolve_vae(None), resolve_text_encoder(None)
            )
        )
        fit = policy.estimate_fit(footprint)  # pure - never mutates the shared policy
        if fit is None:
            return None
        soft = not fit.fits or fit.plan in ("int8", "nf4", "offload")
        return {
            "plan": fit.plan,
            "fits": fit.fits,
            "requiredVramMb": int(fit.required_vram_gb * 1024),
            "totalVramMb": int(fit.total_vram_gb * 1024) if fit.total_vram_gb else None,
            "freeVramMb": policy.free_vram_mb(),
            "freeRamMb": policy.free_ram_mb(),
            "warning": fit.note if soft else None,
        }
