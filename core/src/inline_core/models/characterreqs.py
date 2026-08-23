"""Requirements for the character nodes: the three small encoders encoding and scoring need.

Declared here rather than hardcoded in the client so one answer serves the model popup, a dropped
workflow, and the node itself. Torch-free (pure filesystem), so it registers on a runtime-less
install too - a download only needs the models dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..characters import weights
from ..config import models_dir
from .requirements import ModelComponent

#: Every node that runs an encoder. The rest of the character family only moves bytes around.
ENCODER_NODES = (
    "character/encode", "character/edit", "character/verify-refs", "character/ingest-approved",
)


def _component(
    id_: str, label: str, filename: str, repo: str, present: bool, **kw: Any
) -> ModelComponent:
    # `repo_file` is required and empty for a folder component, so it is defaulted rather than
    # passed at every call site.
    return ModelComponent(
        id=id_,
        label=label,
        category=weights.ANNOTATOR_DIR,
        present=present,
        filename=filename,
        repo=repo,
        repo_file=kw.pop("repo_file", ""),
        **kw,
    )


def encoder_components() -> list[ModelComponent]:
    """The face detector, the face recogniser and DINOv2, with live presence."""
    return [
        _component(
            "yunet",
            "Face detector",
            weights.YUNET_FILE,
            weights.YUNET_REPO,
            weights.yunet_path().is_file(),
            repo_file=weights.YUNET_FILE,
        ),
        _component(
            "sface",
            "Face recognition",
            weights.SFACE_FILE,
            weights.SFACE_REPO,
            weights.sface_path().is_file(),
            repo_file=weights.SFACE_FILE,
        ),
        _component(
            "dinov2",
            "Subject embeddings",
            weights.DINOV2_DIR,
            weights.DINOV2_REPO,
            (weights.dinov2_path() / "model.safetensors").is_file(),
            # Named files rather than the whole repo: DINOv2 ships a .bin twin beside the
            # safetensors. They sit at the repo root, so no `repo_folder` - naming one made the
            # fetch look for a subfolder that is never created.
            repo_files=weights.DINOV2_FILES,
        ),
    ]


class CharacterEncoderProvider:
    """What a character node needs before it can encode or score."""

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return encoder_components()

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        return None
