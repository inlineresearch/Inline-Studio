"""What Z-Image needs on disk, and whether it's present - the data behind the node's model popup.

**No hidden downloads.** A component is "present" only if the user placed it under ``models/`` or
downloaded it through the popup (which also writes into ``models/``). Nothing here or in the runner
ever fetches a model as a side effect of loading - a missing component is reported, not silently
pulled from Hugging Face.

This module is deliberately **torch-free** (pure filesystem + config), so the requirements check and
the download planning work without the heavy ``zimage`` runtime loaded. The runner imports the
resolution helpers from here so the "what/where" logic lives in one place.

Z-Image loads ComfyUI-style from **three single files** - one diffusion ``.safetensors``, one VAE,
one text-encoder - pulled from ``Comfy-Org/z_image/split_files`` and dropped flat into
``diffusion_models/``, ``vae/`` and ``text_encoders/``. The small configs + Qwen tokenizer that
neither weights file carries come from the loader-core asset bundle (``models/loaders.py``), not
here. A whole-pipeline diffusers folder in ``diffusion_models/`` is still accepted as a fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ...config import models_dir

#: Split-file weights repo - ComfyUI's consolidated single files, one ``.safetensors`` per
#: component (fast, fully offline to load). The popup pulls exactly one file per component from
#: here; the configs + tokenizer come from the loader-core asset bundle (see ``models/loaders.py``).
SPLIT_REPO = "Comfy-Org/z_image"
#: Back-compat alias - older callers/tests referenced ``BASE_REPO`` as "the download repo".
BASE_REPO = SPLIT_REPO

#: The exact single files under ``split_files/<category>/`` - mirrors ComfyUI's Z-Image layout. They
#: land flat in ``models/<category>/`` so the node's per-category dropdowns list them directly.
DIFFUSION_FILE = "z_image_bf16.safetensors"
VAE_FILE = "ae.safetensors"
TEXT_ENCODER_FILE = "qwen_3_4b.safetensors"
_SPLIT_PREFIX = "split_files"

_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".sft")
_LOCAL_NAMES = ("Z-Image-Turbo", "z-image-turbo", "Z-Image", "z-image")


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
    for child in sorted(p for p in root.iterdir() if p.is_dir() and not _is_staging(p)):
        if (child / "model_index.json").is_file():
            return child
    return None


def _is_staging(path: Path) -> bool:
    """A half-finished download's ``.part`` staging dir - never treat it as an installed model."""
    return path.name.endswith(".part")


def _category_file(
    category: str, filename: str, env_var: str, chosen: object = None
) -> Path | None:
    """The single weight file the runner loads for a category, or None if the category is empty.

    Resolution, most specific first: an explicit ``env_var`` path, then a ``chosen`` filename from
    the node's dropdown, then the exact recommended split file (e.g. ``ae.safetensors``), then any
    weight file the user dropped in. Files only - the split-file loader needs a single
    ``.safetensors``; a bare HF snapshot dir has nothing to hand to ``from_single_file``.
    """
    env = os.environ.get(env_var, "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    root = models_dir() / category
    if not root.is_dir():
        return None
    chosen_name = str(chosen or "").strip()
    if chosen_name:
        picked = root / chosen_name
        if picked.is_file():
            return picked
    exact = root / filename
    if exact.is_file():
        return exact
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )
    return files[0] if files else None


def resolve_vae(params: dict[str, object] | None = None) -> Path | None:
    """The VAE single file (dropdown pick / ``INLINE_ZIMAGE_VAE`` / split ``ae.safetensors``)."""
    return _category_file("vae", VAE_FILE, "INLINE_ZIMAGE_VAE", (params or {}).get("vae"))


def resolve_text_encoder(params: dict[str, object] | None = None) -> Path | None:
    """The text-encoder single file (dropdown pick / ``INLINE_ZIMAGE_TEXT_ENCODER`` / the split
    ``qwen_3_4b.safetensors``)."""
    chosen = (params or {}).get("text_encoder")
    return _category_file("text_encoders", TEXT_ENCODER_FILE, "INLINE_ZIMAGE_TEXT_ENCODER", chosen)


def resolve_diffusion(params: dict[str, object] | None = None) -> tuple[str, str] | None:
    """Pick the Z-Image diffusion source without ever inventing a remote one.

    Returns ``(mode, path)`` where ``mode`` is ``"single_file"`` (a lone transformer file - VAE and
    text-encoder come from local files) or ``"pipeline"`` (a whole diffusers folder). Returns
    ``None`` when nothing is present locally - reported as missing (no repo-id fallback). Priority:
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
    """One required model component: whether it's present, and the exact file the popup fetches."""

    id: str  # "diffusion" | "vae" | "text_encoder"
    label: str
    category: str  # models/ subfolder it belongs to
    present: bool
    filename: str  # the single file that lands flat in models/<category>/ (a dropdown entry)
    repo: str  # HF repo the popup downloads from
    repo_file: str  # exact repo-relative path fetched from ``repo`` (under ``split_files/``)

    @property
    def local_path(self) -> str:
        """Where this file lives / lands, relative to the models root (flat in its category)."""
        return f"{self.category}/{self.filename}"


def _split_component(
    *, id: str, label: str, category: str, filename: str, present: bool
) -> ModelComponent:
    return ModelComponent(
        id=id,
        label=label,
        category=category,
        present=present,
        filename=filename,
        repo=SPLIT_REPO,
        repo_file=f"{_SPLIT_PREFIX}/{category}/{filename}",
    )


def zimage_requirements(params: dict[str, object] | None = None) -> list[ModelComponent]:
    """The three Z-Image components with live presence, for the node's model popup.

    Presence is the **specific single file** (or any weight the user dropped in that category - see
    ``_category_file``). A whole-pipeline diffusers folder as the diffusion source still counts for
    the VAE + text-encoder, since it already bundles them.
    """
    diffusion = resolve_diffusion(params)
    is_pipeline = diffusion is not None and diffusion[0] == "pipeline"

    return [
        _split_component(
            id="diffusion",
            label="Diffusion model",
            category="diffusion_models",
            filename=DIFFUSION_FILE,
            present=diffusion is not None,
        ),
        _split_component(
            id="vae",
            label="VAE",
            category="vae",
            filename=VAE_FILE,
            present=is_pipeline or resolve_vae(params) is not None,
        ),
        _split_component(
            id="text_encoder",
            label="Text encoder",
            category="text_encoders",
            filename=TEXT_ENCODER_FILE,
            present=is_pipeline or resolve_text_encoder(params) is not None,
        ),
    ]


def download_target(component: ModelComponent) -> Path:
    """Absolute local dir the component's single file lands in - its category folder, flat, so the
    node's dropdown lists it (under the models root, never the hidden HF cache)."""
    return models_dir() / component.category


# --- memory footprint (on-disk sizes, for the device policy's fit estimate) ---------------------


def _file_bytes(path: object) -> int:
    """Size of a single weight file in bytes, or 0 when absent/unreadable (or a folder). The files
    ship already 16-bit, so the size is a good proxy for the fp16-resident weight footprint."""
    text = str(path or "").strip()
    if not text:
        return 0
    try:
        p = Path(text)
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def footprint_bytes(
    diffusion: object = None, vae: object = None, text_encoder: object = None
) -> dict[str, int]:
    """On-disk byte sizes of the three single-file components, keyed to match ``ModelFootprint``'s
    fields. Torch-free (a plain ``stat``) so the policy/UI can size the load without the runtime.
    Pass the already-resolved paths (which honor wired handles); a missing/folder path is 0."""
    return {
        "diffusion_bytes": _file_bytes(diffusion),
        "text_encoder_bytes": _file_bytes(text_encoder),
        "vae_bytes": _file_bytes(vae),
    }
