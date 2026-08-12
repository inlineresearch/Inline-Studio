"""LTX-2.5's answer to the model popup: what it needs, what it would load, and whether it fits.

Torch-free and pure filesystem, like every other provider: this runs on every popup open, including
on an install with no ML stack where the nodes themselves are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent
from . import requirements as reqs


class Ltx25Provider:
    """One instance per node. Every LTX node needs the same files, so ``build`` is the only axis."""

    def __init__(self, build: str = "distilled") -> None:
        self._build = build

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        """The component list, following the node's own mode rather than the default.

        A node set to quality mode needs the dev transformer, and reporting the distilled one as
        required would show "all present" on a machine that cannot render.
        """
        mode = str((params or {}).get("mode") or "")
        return reqs.components("dev" if mode == "quality" else self._build)

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def after_download(self, component: ModelComponent, path: Path) -> None:
        """Remember which file is which build.

        The distilled and dev transformers are the same architecture, the same metadata and the same
        42,018,190,584 bytes, so a renamed file cannot be identified by inspection. This is the only
        record of which is which.
        """
        recorded = {
            "ltx-distilled": "distilled",
            "ltx-dev": "dev",
            "ltx-distilled-nvfp4": "nvfp4",
        }.get(component.id)
        if recorded:
            reqs.record_provenance(recorded, path.name)

    def resolved(self) -> dict[str, str]:
        """What this node would load now, so the pickers show real files rather than "auto"."""
        picks = {
            "model": reqs.resolve_transformer(self._build),
            "text_encoder": reqs.resolve("text_encoders", reqs.TEXT_ENCODER_FILE),
            "vae": reqs.resolve("vae", reqs.VIDEO_VAE_FILE),
            "upscaler": reqs.resolve("latent_upscale_models", reqs.SPATIAL_UPSCALER_FILE),
        }
        return {key: Path(str(value)).name for key, value in picks.items() if value}

    def catalog_options(self, category: str) -> list[str] | None:
        """Only the checkpoints this node can actually load.

        Matched on the safetensors header, not the filename, because `diffusion_models/` is shared
        across architectures and a file can be renamed. The ComfyUI int8 builds are excluded here;
        ``rejected()`` explains why so the UI can say it rather than hiding them.
        """
        if category == "loras":
            return reqs.selectable_loras()
        if category != "diffusion_models":
            return None
        return [path.name for path in reqs.usable_transformers()]

    def rejected(self) -> list[dict[str, str]]:
        """LTX files that are present but unusable, each with the reason."""
        return [
            {"file": candidate.path.name, "reason": candidate.reason}
            for candidate in reqs.rejected_files()
        ]

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether this will fit, before a 71 GB download rather than after.

        Not staged. The pipeline puts the transformer on the card in its constructor and loads Gemma
        lazily at encode time, so both are resident at the peak - measured, after an OOM that said
        so. The estimate therefore sums every component rather than taking the larger half.
        """
        if policy is None:
            return None
        try:
            from ...device.policy import ModelFootprint
        except ImportError:
            return None
        sizes = reqs.footprint_bytes(self._build)
        if not any(sizes.values()):
            return None
        peak = ModelFootprint(
            diffusion_bytes=sizes["diffusion_bytes"],
            text_encoder_bytes=sizes["text_encoder_bytes"],
            vae_bytes=sizes["vae_bytes"],
        )
        fit = policy.estimate_fit(peak)
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
