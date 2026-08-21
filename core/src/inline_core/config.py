"""Engine configuration from the environment. Small and explicit."""

from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """The **writable** models root: where downloads and trained LoRAs land. `INLINE_MODELS_DIR`,
    else `./models`. Users drop their own weight files here; nothing is downloaded."""
    env = os.environ.get("INLINE_MODELS_DIR")
    return Path(env).expanduser() if env else Path("models")


def models_dirs() -> list[Path]:
    """Every root scanned for installed weights, writable one first.

    Pointing `INLINE_MODELS_DIR` (or `--models-dir`) somewhere else used to hide `./models`
    entirely, so a user with weights in both places could only ever see one set. The extra roots
    are read-only: only `models_dir()` is ever written to.
    """
    roots = [models_dir()]
    default = Path("models")
    for extra in (default, *_extra_models_dirs()):
        if extra not in roots and extra.is_dir():
            roots.append(extra)
    return roots


def _extra_models_dirs() -> list[Path]:
    """Additional read-only roots from `INLINE_EXTRA_MODELS_DIRS` (os.pathsep-separated)."""
    raw = os.environ.get("INLINE_EXTRA_MODELS_DIRS", "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def data_dir() -> Path:
    """Engine-owned working data (the run DB, takes). `INLINE_DATA_DIR`, else `./.inline`."""
    env = os.environ.get("INLINE_DATA_DIR")
    return Path(env).expanduser() if env else Path(".inline")


def extensions_dir() -> Path:
    """Community extensions root. `INLINE_EXTENSIONS_DIR`, else `./extensions` (so a dev checkout
    keeps it beside `./models` and `./.inline`). Holds `state.json`, the host constraint snapshot,
    and one directory per installed extension."""
    env = os.environ.get("INLINE_EXTENSIONS_DIR")
    return Path(env).expanduser() if env else Path("extensions")


def models_registry_url() -> str:
    """Where the Models settings page fetches its list. `INLINE_MODEL_REGISTRY`, else the published
    one; point it at another list to see entries the default one does not carry. Download
    coordinates only - what a checkpoint *is* stays a content check."""
    return os.environ.get(
        "INLINE_MODEL_REGISTRY",
        "https://raw.githubusercontent.com/inlineresearch/Inline-Registry/main/models.json",
    )


def registry_url() -> str:
    """Where the Available tab fetches its extension index. `INLINE_EXTENSION_REGISTRY`, else the
    public registry. Point it at a fork or a file:// path to test a registry change."""
    return os.environ.get(
        "INLINE_EXTENSION_REGISTRY",
        "https://raw.githubusercontent.com/inlineresearch/Inline-Registry/main/index.json",
    )


def server_host() -> str:
    """Address the /v1 server binds. `INLINE_HOST`, else loopback (`127.0.0.1`)."""
    return os.environ.get("INLINE_HOST", "127.0.0.1")


def server_port() -> int:
    """Port the /v1 server binds. `INLINE_PORT`, else 8848; a non-numeric value falls back."""
    raw = os.environ.get("INLINE_PORT", "")
    try:
        return int(raw)
    except ValueError:
        return 8848
