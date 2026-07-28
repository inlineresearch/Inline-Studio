"""What Apply ControlNet needs on disk: controlnet_aux's annotator weights, offered as suggested
downloads so the node can prefetch them into ``models/annotators/`` instead of the hidden HF cache.
Torch-free (pure filesystem), so it answers the model popup even on a runtime-less install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import models_dir
from ..requirements import ModelComponent

ANNOTATOR_REPO = "lllyasviel/Annotators"
ANNOTATOR_DIR = "annotators"

# controlnet_aux loads these flat from a local dir: OpenPose needs the body/hand/face files, depth
# needs the MiDaS checkpoint. Canny needs no weights. (id, label, filename).
_ANNOTATORS = (
    ("openpose-body", "OpenPose body", "body_pose_model.pth"),
    ("openpose-hand", "OpenPose hand", "hand_pose_model.pth"),
    ("openpose-face", "OpenPose face", "facenet.pth"),
    ("midas-depth", "MiDaS depth", "dpt_hybrid-midas-501f0c75.pt"),
)


def _present(filename: str) -> bool:
    return (models_dir() / ANNOTATOR_DIR / filename).is_file()


def annotator_components() -> list[ModelComponent]:
    # Suggested (optional): the maps also work by auto-fetching into the HF cache on first run, so a
    # missing detector is a soft suggestion, never a blocking "models missing".
    return [
        ModelComponent(
            id=cid,
            label=label,
            category=ANNOTATOR_DIR,
            present=_present(fn),
            filename=fn,
            repo=ANNOTATOR_REPO,
            repo_file=fn,
            optional=True,
        )
        for cid, label, fn in _ANNOTATORS
    ]


class PreprocessProvider:
    """Apply ControlNet's detector weights, prefetched once into ``models/annotators/``."""

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return annotator_components()

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        return None
