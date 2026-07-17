"""Installed-model catalog and the model runners.

`catalog.py` scans the models root (category subfolders of weight files the user drops in) and feeds
the dynamic `options_from` selects on node params. Runner subpackages (e.g. `zimage`) are optional
and imported best-effort by `server.bootstrap`, so a core install without their deps still boots.
"""

from __future__ import annotations
