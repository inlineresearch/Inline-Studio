"""Z-Image's answer to "what do I need on disk" - the first ``RequirementsProvider``.

The logic is unchanged; it moved here from ``studio/models.py``, where it sat behind a hardcoded
``node_type == "alibaba/z-image-turbo"`` check. Owning it next to the model keeps the Studio
download layer model-agnostic and lets extensions plug in the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent
from .requirements import (
    footprint_bytes,
    resolve_diffusion,
    resolve_text_encoder,
    resolve_vae,
    zimage_requirements,
)


class ZImageProvider:
    """Requirements + fit estimate for ``alibaba/z-image-turbo``."""

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return zimage_requirements(params)

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def resolved(self) -> dict[str, str]:
        """What this node would load right now, so the pickers show real files instead of "auto"."""
        from pathlib import Path as _Path

        diffusion = resolve_diffusion(None)
        picks: dict[str, Any] = {
            "model": diffusion[1] if diffusion and diffusion[0] == "single_file" else None,
            "vae": resolve_vae(None),
            "text_encoder": resolve_text_encoder(None),
        }
        return {k: _Path(str(v)).name for k, v in picks.items() if v}

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether the model will fit this machine, and how (resident / int8 / offload / won't
        fit), so the popup warns BEFORE a load. Pure ``stat`` + a live VRAM/RAM probe; ``None``
        when the runtime/policy is absent or the sizes/device can't be measured (a whole-pipeline
        folder)."""
        if policy is None:
            return None
        try:
            from ...device.policy import ModelFootprint
        except ImportError:
            return None
        diffusion = resolve_diffusion(None)
        diffusion_file = diffusion[1] if diffusion and diffusion[0] == "single_file" else None
        footprint = ModelFootprint(
            **footprint_bytes(diffusion_file, resolve_vae(None), resolve_text_encoder(None))
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
