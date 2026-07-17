"""Engine configuration from the environment. Small and explicit."""

from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """The models root scanned on start (category subfolders inside). `INLINE_MODELS_DIR`, else
    `./models`. Users drop their own weight files here; nothing is downloaded."""
    env = os.environ.get("INLINE_MODELS_DIR")
    return Path(env).expanduser() if env else Path("models")


def data_dir() -> Path:
    """Engine-owned working data (the run DB, takes). `INLINE_DATA_DIR`, else `./.inline`."""
    env = os.environ.get("INLINE_DATA_DIR")
    return Path(env).expanduser() if env else Path(".inline")


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
