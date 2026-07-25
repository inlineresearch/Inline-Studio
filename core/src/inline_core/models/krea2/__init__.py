"""The Krea 2 (Krea AI) runtime: a diffusers-backed text-to-image / img2img runner.

Optional subpackage. `server.bootstrap` imports `register_krea2` best-effort, so a core install
without the ``runtime`` extra (torch + diffusers) still boots and serves the source nodes.
"""

from .runner import register_krea2

__all__ = ["register_krea2"]
