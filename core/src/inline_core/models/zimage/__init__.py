"""The Z-Image (Alibaba Tongyi) runtime: a diffusers-backed text-to-image / img2img runner.

Optional subpackage. `server.bootstrap` imports `register_zimage` best-effort, so a core install
without the ``zimage`` extra (torch + diffusers) still boots and serves the source nodes.
"""

from __future__ import annotations

from .runner import register_zimage

__all__ = ["register_zimage"]
