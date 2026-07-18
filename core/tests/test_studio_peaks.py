"""Waveform peaks: bucketing/normalisation, the id -> source-file mapping, and lazy generation.

The peaks pipeline was lost in the Electron -> Python port, so audio takes rendered with no
waveform. These cover the pure math and the resolution rules without needing ffmpeg; the one test
that actually decodes is skipped when no ffmpeg binary is available.
"""

from __future__ import annotations

import array
import json
import subprocess

import pytest

from inline_core.studio import peaks
from inline_core.studio.store import StudioStore
from inline_core.studio.timeline.ffmpeg import ffmpeg_exe


def _pcm(samples: list[int]) -> bytes:
    return array.array("h", samples).tobytes()


def test_peaks_payload_normalises_and_buckets() -> None:
    # Two buckets: the first peaks at 16384 (0.5), the second at 32767 (~1.0).
    payload = peaks.peaks_payload(_pcm([0, 16384, 0, 32767]), buckets=2)
    assert payload["version"] == peaks.PEAKS_VERSION
    assert payload["peaks"] == [0.5, 1.0]


def test_peaks_payload_uses_absolute_magnitude() -> None:
    """A negative trough is as loud as a positive crest - the envelope must not read as silence."""
    assert peaks.peaks_payload(_pcm([-16384, 0]), buckets=1)["peaks"] == [0.5]
    # -32768 has no positive counterpart; it must clamp rather than overflow past 1.0.
    only = peaks.peaks_payload(_pcm([-32768]), buckets=1)["peaks"][0]
    assert 0.99 <= only <= 1.0


def test_peaks_payload_duration_from_sample_count() -> None:
    """Duration comes from the decoded samples, not ffprobe (which is often absent)."""
    one_second = _pcm([1000] * 8000)  # _SAMPLE_RATE samples
    assert peaks.peaks_payload(one_second, buckets=10)["duration"] == pytest.approx(1.0)


def test_peaks_payload_handles_empty_and_short_input() -> None:
    empty = peaks.peaks_payload(b"", buckets=10)
    assert empty["peaks"] == [] and empty["duration"] == 0.0
    # Fewer samples than buckets must not produce empty/zero-width buckets.
    short = peaks.peaks_payload(_pcm([500, 900, 100]), buckets=50)
    assert len(short["peaks"]) == 3
    # A trailing odd byte (a truncated sample) is ignored rather than raising.
    assert peaks.peaks_payload(_pcm([1000]) + b"\x01", buckets=1)["peaks"] == [pytest.approx(0.0305,
                                                                                            abs=1e-3)]


def test_source_for_peaks_maps_takes_and_assets(tmp_path) -> None:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("wave")
    conn, folder = store.conn(), store.folder()
    (folder / "takes").mkdir(exist_ok=True)
    media = folder / "takes" / "clip.m4a"
    media.write_bytes(b"not-real-audio")

    conn.execute(
        "INSERT INTO takes (id, frame_id, file_path, kind, params, created_at) "
        "VALUES (?,?,?,?,?,?)",
        ("t1", "f1", "takes/clip.m4a", "audio", "{}", 0),
    )
    conn.commit()

    assert peaks.source_for_peaks(conn, folder, "thumbs/take-t1.peaks.json") == media.resolve()
    # The `audio-<id>` convention also resolves against takes (a video's embedded audio).
    assert peaks.source_for_peaks(conn, folder, "thumbs/audio-t1.peaks.json") == media.resolve()
    # Unknown ids and non-peaks paths resolve to nothing.
    assert peaks.source_for_peaks(conn, folder, "thumbs/take-nope.peaks.json") is None
    assert peaks.source_for_peaks(conn, folder, "thumbs/take-t1.png") is None
    assert peaks.source_for_peaks(conn, folder, "thumbs/other-t1.peaks.json") is None


def test_source_for_peaks_resolves_a_frame_id(tmp_path) -> None:
    """The timeline's `sourceId` is a FRAME id for frame/preview/trim sources (what the Trim node
    passes as `audioPeaks`), not a take id - so resolution must fall through to the frame's file."""
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("wave")
    conn, folder = store.conn(), store.folder()
    (folder / "takes").mkdir(exist_ok=True)
    media = folder / "takes" / "song.m4a"
    media.write_bytes(b"not-real-audio")

    conn.execute(
        "INSERT INTO frames "
        "(id, sequence_id, name, kind, position, hero_take_id, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("frame1", "seq1", "Music", "audio", 0, "take1", 0, 0),
    )
    conn.execute(
        "INSERT INTO takes (id, frame_id, file_path, kind, params, created_at) "
        "VALUES (?,?,?,?,?,?)",
        ("take1", "frame1", "takes/song.m4a", "audio", "{}", 0),
    )
    conn.commit()

    # Keyed by the frame id, not the take id - resolves through the frame's hero/latest take.
    assert peaks.source_for_peaks(conn, folder, peaks.audio_peaks_rel("frame1")) == media.resolve()


def test_audio_peaks_rel_matches_the_shared_convention() -> None:
    """Must stay in step with `audioPeaksPath` in src/shared/media.ts."""
    assert peaks.audio_peaks_rel("abc") == "thumbs/audio-abc.peaks.json"


def test_write_peaks_is_best_effort_on_undecodable_input(tmp_path) -> None:
    """A file with no audio stream must not raise - the UI keeps its flat placeholder."""
    src = tmp_path / "junk.m4a"
    src.write_bytes(b"definitely not audio")
    assert peaks.write_peaks(src, tmp_path / "out.peaks.json") is False
    assert not (tmp_path / "out.peaks.json").exists()


@pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg not available")
def test_end_to_end_generates_peaks_from_real_audio(tmp_path) -> None:
    """Decode a real generated tone through ffmpeg and confirm a usable waveform lands on disk."""
    src = tmp_path / "tone.wav"
    exe = ffmpeg_exe()
    assert exe is not None
    # lavfi's `sine` emits at only ~0.125 of full scale, so boost it - that way the peak assertion
    # below actually exercises normalisation instead of passing on a near-silent signal.
    subprocess.run(
        [exe, "-v", "quiet", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-af", "volume=7", "-y", str(src)],
        check=True,
        timeout=60,
    )
    dest = tmp_path / "thumbs" / "take-x.peaks.json"
    assert peaks.write_peaks(src, dest) is True

    data = json.loads(dest.read_text())
    assert data["version"] == peaks.PEAKS_VERSION
    assert data["duration"] == pytest.approx(1.0, abs=0.1)
    assert len(data["peaks"]) > 100
    assert all(0.0 <= p <= 1.0 for p in data["peaks"])
    assert max(data["peaks"]) > 0.5  # a full-scale sine must not read as near-silence
