"""Prebuilt Inline Studio web UI, packaged for Inline Core to serve on one port.

Mirrors ComfyUI's ``comfyui_frontend_package``: the wheel's payload is the built SPA under
``static/`` (index.html + hashed assets). Inline Core resolves this package's ``static/`` dir and
mounts it (see ``inline_core.server.frontend.resolve_frontend_root``), so end users ``pip install``
and get the UI with no Node build.

Populate ``static/`` at publish time with the SPA build:  ``npm run build:spa``  in the Inline Studio
repo, then copy ``dist-web/*`` into ``inline_studio_frontend/static/``.
"""

from __future__ import annotations

from pathlib import Path

#: Absolute path to the built SPA directory (contains index.html once populated at build time).
STATIC_DIR = str(Path(__file__).parent / "static")
