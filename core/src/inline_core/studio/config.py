"""App-global config for the Studio backend: where recents/settings live and where new projects are
created (the browser has no folder picker). Env-driven, mirroring the legacy Node web server.

``INLINE_STUDIO_DATA_DIR``      — app data (recents, settings). Default ``~/.inline-studio-server``
``INLINE_STUDIO_WORKSPACE_DIR`` — where new projects are created. Default ``~/InlineStudioProjects``

The old ``STORYLINE_*`` names still work as deprecated aliases (see CLAUDE.md).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COMFY_URL = os.environ.get("COMFYUI_URL") or "http://127.0.0.1:8188"
DEFAULT_CORE_URL = os.environ.get("INLINE_CORE_URL") or "http://127.0.0.1:8848"


def _env(name: str) -> str | None:
    """Read INLINE_STUDIO_<name>, falling back to the deprecated STORYLINE_<name> alias."""
    return os.environ.get(f"INLINE_STUDIO_{name}") or os.environ.get(f"STORYLINE_{name}")


def data_dir() -> Path:
    return Path(_env("DATA_DIR") or (Path.home() / ".inline-studio-server"))


def workspace_dir() -> Path:
    return Path(_env("WORKSPACE_DIR") or (Path.home() / "InlineStudioProjects"))
