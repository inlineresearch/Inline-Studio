"""Locate the Inline Studio frontend Core serves on its own port (mirrors ComfyUI's frontend pkg).

Resolution order, most specific first:
  1. ``INLINE_FRONTEND_ROOT`` — a local SPA build dir (set directly or via ``main.py
     --front-end-root``); the dev loop — rebuild the UI locally without republishing the package.
  2. the installed ``inline_studio_frontend`` package's ``static/`` dir — the default for end users
     (``pip install`` pulls the built frontend; no Node needed).
  3. ``None`` — Core runs API-only (no UI mounted).

A dir only counts when it actually holds an ``index.html``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _has_index(path: Path) -> bool:
    return (path / "index.html").is_file()


def resolve_frontend_root() -> str | None:
    env = os.environ.get("INLINE_FRONTEND_ROOT", "").strip()
    if env:
        root = Path(env)
        return str(root) if _has_index(root) else None

    try:
        import inline_studio_frontend  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    pkg_file = getattr(inline_studio_frontend, "__file__", None)
    if not pkg_file:
        return None
    static = Path(pkg_file).parent / "static"
    return str(static) if _has_index(static) else None
