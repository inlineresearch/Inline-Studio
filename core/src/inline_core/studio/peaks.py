"""Audio waveform peaks: decode a media file's audio and write the peaks JSON the UI draws.

Restores a capability lost in the Electron -> Python port. The renderer's ``Waveform`` component
fetches ``thumbs/take-<id>.peaks.json`` (see ``src/shared/media.ts``) and silently falls back to a
flat baseline when it 404s - which is why audio takes rendered with no waveform anywhere: the old
``electron/main/media/peaks.ts`` was never reimplemented here.

Peaks are generated **lazily, on first request** (see the ``/media`` route): a miss resolves the id
back to its source file, builds the JSON, and serves it. That is self-healing - audio generated
before this existed gets a waveform too, rather than only newly saved takes - and it keeps the cost
off the generation path, since most takes are never scrubbed.

Only ffmpeg is required (bundled via ``imageio-ffmpeg``); ffprobe is often absent, so the duration
is derived from the decoded sample count instead of a separate probe.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .timeline.ffmpeg import ffmpeg_exe

# Bumped if the JSON shape changes; the client reads `peaks` and ignores unknown fields.
PEAKS_VERSION = 1
# Decode mono at a low rate: peaks only need an amplitude envelope, and 8 kHz keeps even a long
# track's PCM small (~16 KB/s) while giving far more samples than the ~1000 buckets we emit.
_SAMPLE_RATE = 8000
_DEFAULT_BUCKETS = 1000
_MAX_INT16 = 32768.0


def peaks_payload(pcm: bytes, buckets: int = _DEFAULT_BUCKETS) -> dict[str, Any]:
    """Bucket signed 16-bit little-endian mono PCM into ``buckets`` normalised 0..1 peaks.

    Each bucket takes the max absolute sample in its window, so short transients stay visible
    rather than being averaged away. Kept pure and torch/ffmpeg-free so it is directly testable.
    """
    import array

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])  # ignore a trailing odd byte
    total = len(samples)
    duration = total / float(_SAMPLE_RATE) if total else 0.0
    if total == 0 or buckets < 1:
        return {"version": PEAKS_VERSION, "duration": duration, "peaks": []}

    count = min(buckets, total)
    size = total / float(count)
    out: list[float] = []
    for i in range(count):
        start = int(i * size)
        end = int((i + 1) * size) if i < count - 1 else total
        peak = 0
        for j in range(start, max(start + 1, end)):
            value = samples[j]
            # -32768 has no positive counterpart; clamp so abs() can't overflow the range.
            magnitude = 32767 if value <= -32768 else (-value if value < 0 else value)
            if magnitude > peak:
                peak = magnitude
        out.append(round(peak / _MAX_INT16, 4))
    return {"version": PEAKS_VERSION, "duration": round(duration, 3), "peaks": out}


def decode_pcm(src: Path, timeout: int = 120) -> bytes | None:
    """Decode ``src``'s audio to mono 16-bit PCM via ffmpeg. None when ffmpeg is missing, the file
    has no audio stream, or decoding fails - callers then simply skip the waveform."""
    exe = ffmpeg_exe()
    if exe is None or not src.is_file():
        return None
    try:
        proc = subprocess.run(
            [exe, "-v", "quiet", "-i", str(src), "-vn", "-ac", "1",
             "-ar", str(_SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout or None


def write_peaks(src: Path, dest: Path, buckets: int = _DEFAULT_BUCKETS) -> bool:
    """Build ``dest`` (the peaks JSON) from ``src``'s audio. True when written.

    Best-effort by design: a file with no audio stream, a missing ffmpeg, or an unwritable target
    all return False, and the UI keeps its flat-baseline placeholder rather than erroring.
    """
    pcm = decode_pcm(src)
    if not pcm:
        return False
    payload = peaks_payload(pcm, buckets)
    if not payload["peaks"]:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return False
    return True


def audio_peaks_rel(source_id: str) -> str:
    """Project-relative peaks path for a source's audio track. Mirrors ``audioPeaksPath`` in
    ``src/shared/media.ts`` - keep the two in step."""
    return f"thumbs/audio-{source_id}.peaks.json"


def source_for_peaks(conn: Any, folder: Path, rel: str) -> Path | None:
    """Map a requested peaks path back to the media file it describes, or None.

    Two conventions, both from ``src/shared/media.ts``:
      - ``thumbs/take-<takeId>.peaks.json``  - an audio take's own waveform
      - ``thumbs/audio-<id>.peaks.json``     - the audio track of whatever produced the media

    For the ``audio-`` form the id is whatever the timeline calls a ``sourceId``, which is a **frame
    id** for frame/preview/trim sources and an **asset id** for library sources - so a take lookup
    alone is not enough; we fall through takes -> assets -> frames.
    """
    name = Path(rel).name
    if not name.endswith(".peaks.json"):
        return None
    stem = name[: -len(".peaks.json")]
    if stem.startswith("take-"):
        ids, tables, try_frame = stem[len("take-") :], ("takes",), False
    elif stem.startswith("audio-"):
        ids, tables, try_frame = stem[len("audio-") :], ("takes", "assets"), True
    else:
        return None
    if not ids:
        return None
    for table in tables:
        try:
            row = conn.execute(
                f"SELECT file_path FROM {table} WHERE id = ?", (ids,)  # noqa: S608 - fixed literals
            ).fetchone()
        except Exception:  # noqa: BLE001 - a missing table must not break media serving
            continue
        if row and row["file_path"]:
            candidate = (folder / str(row["file_path"])).resolve()
            if candidate.is_file():
                return candidate
    if try_frame:
        # A frame id: resolve through its hero take (imported lazily - frames imports back here).
        try:
            from . import frames as fr

            out = fr.resolve_frame_file(conn, ids)
        except Exception:  # noqa: BLE001
            out = None
        if out and out.get("filePath"):
            candidate = (folder / str(out["filePath"])).resolve()
            if candidate.is_file():
                return candidate
    return None


def ensure_peaks(conn: Any, folder: Path, rel: str) -> Path | None:
    """Resolve + generate the peaks JSON for ``rel`` if it does not exist yet. Returns the file
    when it is available to serve, else None. Safe to call on every miss: once written the
    ``/media`` route serves it directly and never reaches here again."""
    dest = (folder / rel).resolve()
    if dest.is_file():
        return dest
    src = source_for_peaks(conn, folder, rel)
    if src is None:
        return None
    return dest if write_peaks(src, dest) else None
