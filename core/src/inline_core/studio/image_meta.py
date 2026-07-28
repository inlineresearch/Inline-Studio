"""Embed / read the Inline recipe inside a generated PNG's metadata (a tEXt chunk keyed
``inline-studio``), so an exported or shared image is self-describing: dropping it back into Inline
Studio rebuilds the graph that made it. Pillow is a runtime dependency; imports stay inside the
functions so a Pillow-less install degrades to a plain copy rather than failing to import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECIPE_KEY = "inline-studio"


def embed_recipe_png(src: Path, dst: Path, recipe: dict[str, Any]) -> None:
    """Copy ``src`` to ``dst`` as a PNG carrying ``recipe`` in an ``inline-studio`` tEXt chunk."""
    from PIL import Image, PngImagePlugin

    with Image.open(src) as im:
        info = PngImagePlugin.PngInfo()
        info.add_text(RECIPE_KEY, json.dumps(recipe, separators=(",", ":")))
        im.save(dst, format="PNG", pnginfo=info)


def read_recipe_png(path: Path) -> dict[str, Any] | None:
    """The embedded recipe dict, or None when the PNG carries none (or can't be read)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            raw = getattr(im, "text", {}).get(RECIPE_KEY) or (im.info or {}).get(RECIPE_KEY)
        parsed = json.loads(raw) if raw else None
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001 - a bad/foreign PNG simply carries no recipe
        return None
