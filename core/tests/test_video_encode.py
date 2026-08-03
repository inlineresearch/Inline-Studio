"""Video and audio takes: the arg vector, the frame/waveform normalisers, and a real mux.

The round-trip tests shell out to the bundled ffmpeg and then probe the result, because the thing
worth checking is that the file a browser receives actually carries both streams. Everything else is
pure and runs with no ffmpeg at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from inline_core.ffmpeg import ffmpeg_available, ffmpeg_exe
from inline_core.media import MediaKind
from inline_core.runtime.file_store import FileTakeStore
from inline_core.runtime.store import TakeStore
from inline_core.runtime.video_encode import build_encode_args, write_wav

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")


def _frames(count: int = 12, w: int = 64, h: int = 32) -> list[np.ndarray]:
    out = []
    for i in range(count):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[..., 0] = (i * 20) % 256  # moving, so the encode is not all-identical keyframes
        out.append(frame)
    return out


def _tone(seconds: float = 0.5, rate: int = 32000) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    wave = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    return np.stack([wave, wave])  # (channels, samples), which is how a stereo model returns it


def _probe(path: Path) -> dict[str, object]:
    out = subprocess.run(
        [ffmpeg_exe() or "", "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    ).stderr
    return {"video": "Video: h264" in out, "audio": "Audio: aac" in out, "raw": out}


# --- the arg vector (no ffmpeg needed) ----------------------------------------------------------


def test_args_stream_frames_on_stdin_and_target_browser_playback() -> None:
    args = build_encode_args("/tmp/x.mp4", width=64, height=32, fps=24)
    assert args[args.index("-s") + 1] == "64x32"
    assert args[args.index("-i") + 1] == "-"
    assert args[args.index("-pix_fmt") + 1] == "rgb24"  # input format
    assert "libx264" in args and "yuv420p" in args and "+faststart" in args


def test_args_add_a_second_input_and_shortest_only_with_audio() -> None:
    silent = build_encode_args("/tmp/x.mp4", width=64, height=32, fps=24)
    assert "-shortest" not in silent and "aac" not in silent

    scored = build_encode_args("/tmp/x.mp4", width=64, height=32, fps=24, audio_path="/tmp/a.wav")
    assert scored.count("-i") == 2 and "/tmp/a.wav" in scored
    assert "-shortest" in scored and "aac" in scored


def test_args_pad_only_an_odd_canvas() -> None:
    assert "-vf" not in build_encode_args("/tmp/x.mp4", width=64, height=32, fps=24)
    odd = build_encode_args("/tmp/x.mp4", width=63, height=32, fps=24)
    assert odd[odd.index("-vf") + 1].startswith("pad=")


# --- the waveform normaliser (no ffmpeg needed) -------------------------------------------------


def test_write_wav_accepts_channels_first_stereo(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "a.wav"
    write_wav(path, _tone(), 32000)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2  # (2, samples) transposed, not read as 2 long samples
        assert handle.getframerate() == 32000
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 16000


def test_write_wav_accepts_mono(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "m.wav"
    write_wav(path, np.zeros(800, dtype=np.float32), 16000)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1 and handle.getnframes() == 800


# --- real encodes -------------------------------------------------------------------------------


@needs_ffmpeg
def test_save_video_writes_one_mp4_carrying_both_streams(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)

    take = store.save_video(
        "run1", "n1", _frames(), {"seed": 3}, fps=24, audio=_tone(), sample_rate=32000
    )

    assert take.kind is MediaKind.VIDEO
    assert take.hash.startswith("sha256-")
    assert take.params["seed"] == 3
    path = Path(take.uri)
    assert path.suffix == ".mp4" and path.stat().st_size > 0
    probed = _probe(path)
    assert probed["video"], probed["raw"]
    assert probed["audio"], probed["raw"]
    # The temporary wav handed to ffmpeg must not survive beside the take.
    assert not list(tmp_path.glob("*.audio.wav"))


@needs_ffmpeg
def test_save_video_without_audio_has_no_audio_stream(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)
    take = store.save_video("run1", "n1", _frames(), {}, fps=24)
    probed = _probe(Path(take.uri))
    assert probed["video"] and not probed["audio"]


@needs_ffmpeg
def test_save_video_accepts_a_float_chw_stack(tmp_path: Path) -> None:
    """What a torch pipeline hands back: (T, C, H, W) floats in 0..1, not a list of PIL images."""
    store = FileTakeStore(tmp_path)
    stack = np.zeros((8, 3, 32, 64), dtype=np.float32)
    stack[:, 1] = 0.5

    take = store.save_video("run1", "n1", stack, {}, fps=24)

    assert Path(take.uri).stat().st_size > 0
    assert _probe(Path(take.uri))["video"]


@needs_ffmpeg
def test_save_audio_writes_a_wav_take(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)
    take = store.save_audio("run1", "n1", _tone(), {}, sample_rate=32000)
    assert take.kind is MediaKind.AUDIO
    assert Path(take.uri).suffix == ".wav" and Path(take.uri).stat().st_size > 0


def test_a_store_that_only_does_images_says_so_rather_than_crashing() -> None:
    """Extensions may supply their own store; reaching video on one must name the store."""

    class ImagesOnly(TakeStore):
        def save(self, run_id, node_id, image, params):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    with pytest.raises(NotImplementedError, match="ImagesOnly"):
        ImagesOnly().save_video("r", "n", [], {}, fps=24)
