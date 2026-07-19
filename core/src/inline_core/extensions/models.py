"""Manifest-declared weights, as a ``RequirementsProvider``.

Turns manifest data into the interface Z-Image implements, so extension models reuse the existing
model popup, download events, and ``options_from`` dropdowns with no frontend work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import models_dir
from ..models.catalog import CATEGORIES
from ..models.requirements import ModelComponent
from .manifest import ModelRequirement


class ManifestRequirements:
    """Requirements for a node whose weights are declared in the manifest."""

    def __init__(self, requirements: tuple[ModelRequirement, ...]) -> None:
        self._requirements = requirements

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        root = models_dir()
        out: list[ModelComponent] = []
        for req in self._requirements:
            category = _checked_category(req.category)
            filename = req.filename or Path(req.repo_file).name
            out.append(
                ModelComponent(
                    id=req.id,
                    label=req.label,
                    category=category,
                    present=(root / category / filename).exists(),
                    filename=filename,
                    repo=req.repo,
                    repo_file=req.repo_file,
                )
            )
        return out

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / _checked_category(component.category)

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """None in v1: the manifest carries no footprint data, and a wrong "this will fit" is worse
        than no estimate."""
        return None


def _checked_category(category: str) -> str:
    """Confine downloads to a known models subfolder. Manifest validation rejects bad categories
    first, so reaching here means a hand-edited install - raise rather than guess a substitute."""
    cleaned = category.strip().strip("/")
    if cleaned not in CATEGORIES:
        raise ValueError(f"unknown model category {category!r}")
    return cleaned
