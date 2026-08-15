"""The three small models character encoding and scoring need.

All permissively licensed and ungated, because the default path must not pull a non-commercial
weight; see ``docs/characters.md``. They land in ``models/annotators/`` so the Models panel shows
them, fetched once on an explicit user action rather than during a render.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from ..config import models_dir
from ..errors import ComponentError

ANNOTATOR_DIR = "annotators"

#: Bumped when a weight file changes, which invalidates every centroid it produced. Cosine
#: similarity across two encoder builds is a number with no meaning, so this is not cosmetic.
YUNET_VERSION = "2023mar"
SFACE_VERSION = "2021dec"
DINOV2_VERSION = "1"

YUNET_REPO = "opencv/face_detection_yunet"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

SFACE_REPO = "opencv/face_recognition_sface"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

DINOV2_REPO = "facebook/dinov2-base"
DINOV2_DIR = "dinov2-base"
DINOV2_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

_LOCK = Lock()


def annotators_root() -> Path:
    return models_dir() / ANNOTATOR_DIR


def yunet_path() -> Path:
    return annotators_root() / YUNET_FILE


def sface_path() -> Path:
    return annotators_root() / SFACE_FILE


def dinov2_path() -> Path:
    return annotators_root() / DINOV2_DIR


def present() -> bool:
    """Whether every scoring weight is already on disk, so a caller can skip the fetch entirely."""
    return (
        yunet_path().is_file()
        and sface_path().is_file()
        and (dinov2_path() / "model.safetensors").is_file()
    )


def ensure() -> None:
    """Fetch whatever is missing (~370 MB, once). Raises ComponentError with actionable copy."""
    if present():
        return
    with _LOCK:
        if present():
            return
        root = annotators_root()
        root.mkdir(parents=True, exist_ok=True)
        try:
            if not yunet_path().is_file():
                _fetch(YUNET_REPO, YUNET_FILE, root)
            if not sface_path().is_file():
                _fetch(SFACE_REPO, SFACE_FILE, root)
            if not (dinov2_path() / "model.safetensors").is_file():
                for name in DINOV2_FILES:
                    _fetch(DINOV2_REPO, name, dinov2_path())
        except Exception as error:  # noqa: BLE001 - re-raised as clear, actionable copy
            raise ComponentError(
                f"Could not fetch the character scoring models ({error}). This is a one-time "
                "~370 MB download into models/annotators/; after it, characters work offline."
            ) from error


def _fetch(repo: str, path: str, root: Path) -> str:
    """Ambient token first, anonymous second - a stale token 401s even on a public repo."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    root.mkdir(parents=True, exist_ok=True)
    try:
        return hf_hub_download(repo, path, local_dir=str(root))
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status not in (401, 403):
            raise
        return hf_hub_download(repo, path, local_dir=str(root), token=False)
