"""Krea 2's answer to "what do I need on disk" - one provider per node variant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent
from .requirements import (
    footprint_bytes,
    krea2_requirements,
    resolve_diffusion,
    resolve_text_encoder,
    resolve_vae,
)


class Krea2Provider:
    """Requirements + fit estimate for one Krea 2 node (``turbo`` or ``raw``)."""

    def __init__(self, variant: str) -> None:
        self._variant = variant

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return krea2_requirements(self._variant, params)

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def resolved(self) -> dict[str, str]:
        """What this node would load right now, so the pickers show real files instead of "auto"."""
        from pathlib import Path as _Path

        picks = {
            "model": resolve_diffusion(self._variant),
            "vae": resolve_vae(None),
            "text_encoder": resolve_text_encoder(None),
        }
        return {k: _Path(str(v)).name for k, v in picks.items() if v}

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether the model will fit this machine, so the popup can warn before a 26GB load.
        ``None`` whenever it can't be sized - a wrong estimate is worse than none."""
        if policy is None:
            return None
        try:
            from ...device.policy import ModelFootprint
        except ImportError:
            return None
        footprint = ModelFootprint(
            **footprint_bytes(
                resolve_diffusion(self._variant), resolve_vae(None), resolve_text_encoder(None)
            )
        )
        fit = policy.estimate_fit(footprint)  # pure - never mutates the shared policy
        if fit is None:
            return None
        soft = not fit.fits or fit.plan in ("int8", "offload")
        return {
            "plan": fit.plan,
            "fits": fit.fits,
            "requiredVramMb": int(fit.required_vram_gb * 1024),
            "totalVramMb": int(fit.total_vram_gb * 1024) if fit.total_vram_gb else None,
            "freeVramMb": policy.free_vram_mb(),
            "freeRamMb": policy.free_ram_mb(),
            "warning": fit.note if soft else None,
        }
