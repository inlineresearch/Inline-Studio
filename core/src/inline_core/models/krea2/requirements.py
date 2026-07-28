"""What Krea 2 needs on disk, and whether it's present - the data behind the node's model popup.

**No hidden downloads**, same posture as Z-Image: a component is present only because the user
placed it under ``models/`` or fetched it from the popup. Torch-free (pure filesystem), so the popup
works on an install with no ML stack.

Krea 2 ships as two checkpoints sharing every other component: RAW for fine-tuning, Turbo for 8-step
inference. The diffusion + text-encoder files are ComfyUI's, from ``Comfy-Org/Krea-2``. The **VAE is
the exception**: ComfyUI's ``qwen_image_vae.safetensors`` uses a layout diffusers cannot read and
there is no converter, so the popup fetches the diffusers-format file from ``Qwen/Qwen-Image``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...config import models_dir
from ..requirements import ModelComponent

#: ComfyUI's repackaged Krea 2 weights. Only the bf16 builds are loadable here - the fp8 / int8 /
#: nvfp4 files carry ComfyUI-specific scale tensors (see `convert.is_quantized_checkpoint`).
COMFY_REPO = "Comfy-Org/Krea-2"
#: The Qwen-Image VAE in diffusers layout, which ComfyUI's copy of the same weights is not.
VAE_REPO = "Qwen/Qwen-Image"
VAE_FILE = "qwen_image_vae_diffusers.safetensors"
VAE_REPO_FILE = "vae/diffusion_pytorch_model.safetensors"
TEXT_ENCODER_FILE = "qwen3vl_4b_bf16.safetensors"

#: variant -> the file the popup downloads for that node.
DIFFUSION_FILES = {
    "turbo": "krea2_turbo_bf16.safetensors",
    "raw": "krea2_raw_bf16.safetensors",
}

VARIANTS = tuple(DIFFUSION_FILES)

_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".sft")
_ENV = {
    "diffusion": "INLINE_KREA2_MODEL",
    "vae": "INLINE_KREA2_VAE",
    "text_encoder": "INLINE_KREA2_TEXT_ENCODER",
}


# --- filesystem resolution (shared with the runner) ---------------------------------------------


def _weight_files(category: str) -> list[Path]:
    root = models_dir() / category
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES)


def _env_path(kind: str) -> Path | None:
    value = os.environ.get(_ENV[kind], "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def foreign_model_message(path: str) -> str | None:
    """A clear error when a diffusion file plainly belongs to another architecture (a Z-Image file
    picked for a Krea 2 node) - name-based, best-effort, to avoid silently distorted output."""
    name = Path(path).name.lower()
    if ("z_image" in name or "z-image" in name) and "krea" not in name:
        return (
            f"'{Path(path).name}' is a Z-Image model, but this is a Krea 2 node. Pick a "
            "krea2_*.safetensors in the Diffusion file dropdown (or clear it to auto-select)."
        )
    return None


def resolve_diffusion(variant: str, params: dict[str, object] | None = None) -> Path | None:
    """The Krea 2 transformer file for this node: the dropdown pick, the env override, the exact
    recommended file, else any krea2 file matching the variant. A user holding both RAW and Turbo
    must get the one their node asks for, so a name that mentions the *other* variant is skipped."""
    chosen = str((params or {}).get("model") or "").strip()
    if chosen:
        picked = models_dir() / "diffusion_models" / chosen
        if picked.is_file():
            return picked
    env = _env_path("diffusion")
    if env is not None:
        return env

    files = _weight_files("diffusion_models")
    exact = models_dir() / "diffusion_models" / DIFFUSION_FILES[variant]
    if exact.is_file():
        return exact
    other = [v for v in VARIANTS if v != variant]
    krea = [
        p for p in files
        if "krea" in p.name.lower() and not any(o in p.name.lower() for o in other)
    ]
    return krea[0] if krea else None


def resolve_vae(params: dict[str, object] | None = None) -> Path | None:
    return _resolve_shared("vae", "vae", VAE_FILE, params)


def resolve_text_encoder(params: dict[str, object] | None = None) -> Path | None:
    return _resolve_shared("text_encoder", "text_encoders", TEXT_ENCODER_FILE, params)


def _resolve_shared(
    kind: str, category: str, filename: str, params: dict[str, object] | None
) -> Path | None:
    """A component both Krea 2 nodes share. Unlike Z-Image this never falls back to "any file in the
    folder": the folders also hold Z-Image's VAE and encoder, and picking one of those would fail
    deep inside the load rather than in the popup."""
    chosen = str((params or {}).get(kind) or "").strip()
    if chosen:
        picked = models_dir() / category / chosen
        if picked.is_file():
            return picked
    env = _env_path(kind)
    if env is not None:
        return env
    exact = models_dir() / category / filename
    return exact if exact.is_file() else None


# --- the requirements view (the popup's data) ---------------------------------------------------


def krea2_requirements(
    variant: str, params: dict[str, object] | None = None
) -> list[ModelComponent]:
    """The three Krea 2 components with live presence, for the node's model popup."""
    diffusion_file = DIFFUSION_FILES[variant]
    return [
        ModelComponent(
            id="diffusion",
            label=f"Diffusion model ({variant.upper()})",
            category="diffusion_models",
            present=resolve_diffusion(variant, params) is not None,
            filename=diffusion_file,
            repo=COMFY_REPO,
            repo_file=f"diffusion_models/{diffusion_file}",
        ),
        ModelComponent(
            id="text_encoder",
            label="Text encoder (Qwen3-VL 4B)",
            category="text_encoders",
            present=resolve_text_encoder(params) is not None,
            filename=TEXT_ENCODER_FILE,
            repo=COMFY_REPO,
            repo_file=f"text_encoders/{TEXT_ENCODER_FILE}",
        ),
        ModelComponent(
            id="vae",
            label="VAE (Qwen-Image)",
            category="vae",
            present=resolve_vae(params) is not None,
            filename=VAE_FILE,
            repo=VAE_REPO,
            repo_file=VAE_REPO_FILE,
        ),
    ]


def download_target(component: ModelComponent) -> Path:
    return models_dir() / component.category


# --- memory footprint (on-disk sizes, for the device policy's fit estimate) ---------------------


def _file_bytes(path: object) -> int:
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
    """On-disk sizes keyed to match ``ModelFootprint``. Torch-free (a plain ``stat``)."""
    return {
        "diffusion_bytes": _file_bytes(diffusion),
        "text_encoder_bytes": _file_bytes(text_encoder),
        "vae_bytes": _file_bytes(vae),
    }
