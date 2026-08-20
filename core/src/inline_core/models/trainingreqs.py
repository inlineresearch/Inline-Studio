"""What Train LoRA needs on disk, read off the architecture and base its own settings pick.

The base checkpoint is the one model a training graph never names. The engine resolves it at run
time from the architecture against what is installed (``training/models.py``), so an exported graph
listed the character encoders and left a 26 GB checkpoint for the reader to work out.

Torch-free like every provider: each architecture's own requirements module already answers "which
files, and are they here", and this only chooses which build to ask it about.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import models_dir
from .requirements import ModelComponent

#: The node types this provider answers for.
TRAINING_NODES = ("train/lora",)


def _hyperparams(params: dict[str, Any] | None) -> dict[str, Any]:
    raw = (params or {}).get("hyperparams")
    return raw if isinstance(raw, dict) else {}


def base_components(arch: str, base_mode: str) -> list[ModelComponent]:
    """The required components for one architecture's training base, newest-arch-first by name."""
    if arch == "krea2":
        from .krea2 import requirements as reqs

        # RAW is the fine-tuning build; Turbo only when the run adds the de-distillation adapter.
        variant = "turbo" if base_mode == "turbo_adapter" else "raw"
        return _required(reqs.krea2_requirements(variant)) + _adapter(arch, base_mode)
    if arch == "z-image":
        from .zimage import requirements as reqs

        return _required(reqs.zimage_requirements()) + _adapter(arch, base_mode)
    if arch == "flux2":
        from .flux2 import requirements as reqs

        # The distilled build is what the generation node wants and what the trainer refuses, so
        # the Base checkpoint FLUX.2 lists as an optional extra is the required one here.
        rows = reqs.flux2_requirements()
        base = next((c for c in rows if c.id == "diffusion_klein_4b_base"), None)
        keep = [c for c in _required(rows) if c.id != "diffusion"]
        return ([replace(base, optional=False)] if base else _required(rows)) + keep
    if arch == "ltx-2-5":
        from .ltx25 import requirements as reqs

        # dev is the only LTX build that trains; the distilled one is refused by the trainer.
        return _required(reqs.components("dev"))
    if arch == "minimax-h3":
        from .minimaxh3 import requirements as reqs

        return _required(reqs.components())
    return []


def _required(components: list[ModelComponent]) -> list[ModelComponent]:
    """Suggested extras belong to generation, not to a training run's pre-flight."""
    return [c for c in components if not c.optional]


#: The de-distillation adapter each arch needs to train against its Turbo build without drift.
#: A generation node never loads one, so no model popup offered it and the run failed at the point
#: of no return with a repo name and nothing to click.
_TURBO_ADAPTERS: dict[str, tuple[str, str]] = {
    "krea2": ("ostris/krea2_turbo_training_adapter", "krea2_turbo_training_adapter_v1.safetensors"),
    "z-image": (
        "ostris/zimage_turbo_training_adapter",
        "zimage_turbo_training_adapter_v2.safetensors",
    ),
}


def _adapter(arch: str, base_mode: str) -> list[ModelComponent]:
    """The training adapter, required only in Turbo mode. Empty for every other base."""
    if base_mode != "turbo_adapter":
        return []
    entry = _TURBO_ADAPTERS.get(arch)
    if entry is None:
        return []
    repo, filename = entry
    return [
        ModelComponent(
            id="training_adapter",
            label="Turbo training adapter (de-distillation)",
            category="loras",
            present=(models_dir() / "loras" / filename).is_file(),
            filename=filename,
            repo=repo,
            repo_file=filename,
        )
    ]


class TrainingBaseProvider:
    """Answers the Train LoRA node for whichever architecture its settings select."""

    def components(self, params: dict[str, Any] | None = None) -> list[ModelComponent]:
        hyper = _hyperparams(params)
        return base_components(str(hyper.get("arch") or ""), str(hyper.get("baseMode") or ""))

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category
