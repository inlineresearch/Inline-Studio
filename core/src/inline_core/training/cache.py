"""Build the training cache: every latent and every conditioning tensor, before the base loads.

One seam so ``trainer.py`` stays a single loop. Most architectures load their VAE and text encoder
together, encode in one sweep and drop both; MiniMax H3 cannot hold those two at once and stages
them (see ``h3.py``). Either way the transformer loads into a card with nothing else on it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import arch as archs
from . import dataset as ds
from . import models
from . import precache_store as store

#: MiniMax H3's video sigma shift. Its scheduler applies the same expression Z-Image's does, so the
#: arch samples through it unchanged.
_H3_SHIFT = 12.0


def build(
    dataset_dir: str,
    models_dir: str,
    arch: str,
    device: str,
    dtype: Any,
    resolution: int,
    *,
    flip: bool = False,
    dropout: float = 0.0,
    clip_frames: int = 1,
    clip_window: str = "start",
    cache_dir: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    """Return ``(items, unconditional, shift)``, all as CPU tensors, with the encoders freed.

    ``on_status`` reports phase progress to the caller, which forwards it over the JSON protocol.
    Precaching a large dataset takes minutes, and a logger call would be dropped here: the trainer
    subprocess configures no logging handler, so anything below WARNING goes nowhere.

    ``cache_dir`` keeps the encode on disk between runs. Without it a resume re-encodes everything,
    which on a card that cannot hold the text conditioner means hours of CPU work to reach the step
    it left off at.
    """
    settings = {
        "resolution": resolution, "flip": flip, "dropout": dropout > 0,
        "clip_frames": clip_frames, "clip_window": clip_window,
    }
    say = on_status or (lambda _text: None)
    key = None
    if cache_dir:
        say("checking for a cached dataset")
        key = store.fingerprint(dataset_dir, arch, settings)
        hit = store.load(Path(cache_dir), key)
        if hit is not None:
            items, unconditional, shift = hit
            say(f"reusing {len(items)} cached items, nothing to encode")
            return items, unconditional, shift

    items, unconditional, shift = _encode(
        dataset_dir, models_dir, arch, device, dtype, resolution,
        flip=flip, dropout=dropout, clip_frames=clip_frames, clip_window=clip_window,
        on_status=on_status,
    )
    if cache_dir and key is not None:
        store.save(Path(cache_dir), key, items, unconditional, shift)
    return items, unconditional, shift


def _encode(
    dataset_dir: str,
    models_dir: str,
    arch: str,
    device: str,
    dtype: Any,
    resolution: int,
    *,
    flip: bool = False,
    dropout: float = 0.0,
    clip_frames: int = 1,
    clip_window: str = "start",
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    if arch == archs.MINIMAX_H3:
        from . import h3

        items, unconditional = h3.precache(
            dataset_dir, models_dir, device, dtype, resolution, flip, dropout > 0, clip_frames,
            clip_window=clip_window, on_status=on_status,
        )
        return items, unconditional, _H3_SHIFT

    encoders = models.load_encoders(models_dir, arch, device, dtype)
    items = ds.precache(dataset_dir, encoders, arch, device, dtype, resolution, flip=flip)
    unconditional = ds.precache_empty(encoders, arch, device) if dropout > 0 else None
    shift = float(encoders.scheduler.config.get("shift", 1.0) or 1.0)
    models.free_encoders(encoders)
    return items, unconditional, shift
