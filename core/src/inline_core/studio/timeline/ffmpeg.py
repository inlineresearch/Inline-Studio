"""ffmpeg/ffprobe for the timeline: probe media and run a render with progress.

Binary lookup moved to ``inline_core.ffmpeg`` (the take store needs it too) and is re-exported here
so timeline callers keep importing it from this module.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable

from ...ffmpeg import ffmpeg_available, ffmpeg_exe, ffprobe_exe

__all__ = ["compose_render", "ffmpeg_available", "ffmpeg_exe", "ffprobe_exe", "probe_media"]


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
