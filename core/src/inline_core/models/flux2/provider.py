"""FLUX.2's answer to "what do I need on disk" - the model popup's data source for the node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent
from .requirements import (
    flux2_checkpoints,
    flux2_encoders,
    flux2_requirements,
    footprint_bytes,
    resolve_diffusion,
    resolve_text_encoder,
    resolve_vae,
    resolved_variant,
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

    def resolved(self) -> dict[str, str]:
        """What the node would load right now, so its pickers open on the real files rather than
        on "auto". Names are relative to their category folder, matching the dropdown values."""
        picks = {
            "model": resolve_diffusion(None),
            "vae": resolve_vae(None),
            "text_encoder": resolve_text_encoder(None),
        }
        out = {key: path.name for key, path in picks.items() if path is not None}
        variant = resolved_variant(None)
        if variant is not None:
            out["variant"] = variant.key
        return out

    def catalog_options(self, category: str) -> list[str] | None:
        """Only the files this node can actually load. The categories are shared with Z-Image and
        Krea 2, so an unfiltered list offers checkpoints that would fail on load."""
        if category == "diffusion_models":
            return [p.name for p in flux2_checkpoints()]
        if category == "text_encoders":
            return [p.name for p in flux2_encoders()]
        return None

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
