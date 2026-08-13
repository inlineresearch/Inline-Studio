"""LTX-2.5: text, image and keyframe pairs to video with a synchronised soundtrack.

The runner and its descriptors are re-exported here; `pipeline` and `vendor` are imported lazily so
this package stays importable for the model popup on an install with no ML stack.
"""

from __future__ import annotations

from .runner import DESCRIPTORS, GRID, VARIANTS, Ltx25Runner, Variant, register_ltx25

__all__ = [
    "DESCRIPTORS",
    "GRID",
    "VARIANTS",
    "Ltx25Runner",
    "Variant",
    "register_ltx25",
]
