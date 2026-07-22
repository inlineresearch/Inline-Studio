"""Locate the ffmpeg/ffprobe binaries. Prefers a bundled ``imageio-ffmpeg``, else PATH.

Lives at the top level rather than under ``studio/`` because both the timeline (studio) and the
take store (runtime) need it, and runtime must not import studio.
"""

from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def ffprobe_exe() -> str | None:
    """PATH only - imageio bundles ffmpeg alone, so probing degrades gracefully when absent."""
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None
