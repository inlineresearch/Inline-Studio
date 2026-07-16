"""What Z-Image needs on disk, and whether it's present — the data behind the node's model popup.

**No hidden downloads.** A component is "present" only if the user placed it under ``models/`` or
downloaded it through the popup (which also writes into ``models/``). Nothing here or in the runner
ever fetches a model as a side effect of loading — a missing component is reported, not silently
pulled from Hugging Face.

This module is deliberately **torch-free** (pure filesystem + config), so the requirements check and
the download planning work without the heavy ``zimage`` runtime loaded. The runner imports the
resolution helpers from here so the "what/where" logic lives in one place.

Z-Image is a single diffusers repo (``Tongyi-MAI/Z-Image-Turbo``) with ``transformer/``, ``vae/``,
``text_encoder/`` and ``tokenizer/`` subfolders. The user's common path is to drop a single
diffusion ``.safetensors`` in ``diffusion_models/`` and download the small VAE + text-encoder; the
popup can also fetch the whole pipeline as one diffusers folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...config import models_dir

#: The reference repo. Used only as an explicit download source (never as an implicit load source).
BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"

_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".sft")
_LOCAL_NAMES = ("Z-Image-Turbo", "z-image-turbo", "Z-Image", "z-image")
#: Folder name new downloads land under, inside each category dir (so re-downloads are idempotent).
_LOCAL_DIR = "z-image-turbo"


# --- filesystem resolution (shared with the runner) ---------------------------------------------


def diffusion_root() -> Path:
    return models_dir() / "diffusion_models"


def find_weight_file(root: Path) -> Path | None:
    """The single diffusion weight file to load: prefer a z-image-named file, else the first one."""
    if not root.is_dir():
        return None
    weights = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )
    named = [p for p in weights if "z" in p.name.lower() and "image" in p.name.lower()]
    return (named or weights or [None])[0]


def pipeline_dir(root: Path) -> Path | None:
    """A local diffusers folder (has ``model_index.json``) that holds the whole Z-Image pipeline."""
    if not root.is_dir():
        return None
    for name in _LOCAL_NAMES:
        candidate = root / name
        if (candidate / "model_index.json").is_file():
            return candidate
    # Any subfolder that looks like a diffusers pipeline also counts (e.g. a popup download).
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "model_index.json").is_file():
            return child
    return None


def local_component(category: str, env_var: str) -> Path | None:
    """A local supporting-model file/dir under ``models/<category>/`` (or an env override), or None.
    Prefers an explicit env path, then a single weight file, then a subdir (HF snapshot)."""
    env = os.environ.get(env_var, "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    root = models_dir() / category
    if not root.is_dir():
        return None
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )
    if files:
        return files[0]
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    return dirs[0] if dirs else None


def resolve_diffusion(params: dict[str, object] | None = None) -> tuple[str, str] | None:
    """Pick the Z-Image diffusion source without ever inventing a remote one.

    Returns ``(mode, path)`` where ``mode`` is ``"single_file"`` (a lone transformer file — VAE and
    text-encoder come from local files) or ``"pipeline"`` (a whole diffusers folder). Returns
    ``None`` when nothing is present locally — reported as missing (no repo-id fallback). Priority:
    node ``model`` param, ``INLINE_ZIMAGE_MODEL`` env, a single weight file, then a diffusers dir.
    """
    root = diffusion_root()

    chosen = str((params or {}).get("model") or "").strip()
    if chosen:
        path = root / chosen
        if path.is_file():
            return "single_file", str(path)

    env = os.environ.get("INLINE_ZIMAGE_MODEL", "").strip()
    if env:
        # An explicit override is trusted: a file is a single-file source, anything else a pipeline.
        return ("single_file", env) if Path(env).is_file() else ("pipeline", env)

    single = find_weight_file(root)
    if single is not None:
        return "single_file", str(single)

    pipe = pipeline_dir(root)
    if pipe is not None:
        return "pipeline", str(pipe)

    return None


# --- the requirements view (the popup's data) ---------------------------------------------------


@dataclass(frozen=True)
class ModelComponent:
    """One required model component, whether it's present, and how the popup would download it."""

    id: str  # "diffusion" | "vae" | "text_encoder"
    label: str
    category: str  # models/ subfolder it belongs to
    present: bool
    local_path: str  # where it's expected / would land (relative to the models root)
    repo: str  # HF repo the popup downloads from
    subfolders: tuple[str, ...]  # repo subfolders to fetch & flatten; () = whole repo, keep layout


def zimage_requirements(params: dict[str, object] | None = None) -> list[ModelComponent]:
    """The three Z-Image components with live presence, for the node's model popup.

    VAE and text-encoder count as present when the diffusion source is a whole-pipeline folder (it
    already contains them) or when a local file/dir is provided under their category.
    """
    diffusion = resolve_diffusion(params)
    is_pipeline = diffusion is not None and diffusion[0] == "pipeline"

    vae_local = local_component("vae", "INLINE_ZIMAGE_VAE")
    te_local = local_component("text_encoders", "INLINE_ZIMAGE_TEXT_ENCODER")

    return [
        ModelComponent(
            id="diffusion",
            label="Diffusion model",
            category="diffusion_models",
            present=diffusion is not None,
            local_path=f"diffusion_models/{_LOCAL_DIR}",
            repo=BASE_REPO,
            subfolders=(),  # the whole diffusers pipeline as one folder
        ),
        ModelComponent(
            id="vae",
            label="VAE",
            category="vae",
            present=is_pipeline or vae_local is not None,
            local_path=f"vae/{_LOCAL_DIR}",
            repo=BASE_REPO,
            subfolders=("vae",),
        ),
        ModelComponent(
            id="text_encoder",
            label="Text encoder",
            category="text_encoders",
            present=is_pipeline or (te_local is not None and te_local.is_dir()),
            local_path=f"text_encoders/{_LOCAL_DIR}",
            repo=BASE_REPO,
            subfolders=("text_encoder", "tokenizer"),
        ),
    ]


def download_target(component: ModelComponent) -> Path:
    """Absolute local dir a component downloads into (under the models root, never the HF cache)."""
    return models_dir() / component.local_path
