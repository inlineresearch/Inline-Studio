"""ffmpeg/ffprobe for the timeline: locate the binary, probe media, run a render with progress.

Prefers a bundled ``imageio-ffmpeg`` binary, else a system ``ffmpeg`` on PATH. ffprobe comes from
PATH only (imageio bundles ffmpeg alone); probing degrades gracefully when it's absent.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Callable
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
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None


def probe_media(abs_path: str) -> dict[str, object]:
    """``{"durationSec", "hasAudio"}`` via ffprobe; conservative defaults if unavailable."""
    probe = ffprobe_exe()
    if probe:
        try:
            out = subprocess.run(
                [probe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
                 abs_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(out.stdout or "{}")
            duration = float(data.get("format", {}).get("duration") or 0)
            has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
            return {"durationSec": duration, "hasAudio": has_audio}
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
            pass
    return {"durationSec": 0.0, "hasAudio": False}


async def compose_render(
    args: list[str], total: float, on_progress: Callable[[float], None]
) -> bool:
    """Run ffmpeg with the arg vector, parsing -progress for a 0..1 fraction. True on success."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg is not available.")
    proc = await asyncio.create_subprocess_exec(
        exe, "-progress", "pipe:1", "-nostats", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="ignore").strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=", 1)[1])
                on_progress(min(1.0, (ms / 1_000_000) / max(0.04, total)))
            except ValueError:
                pass
    await proc.wait()
    return proc.returncode == 0
