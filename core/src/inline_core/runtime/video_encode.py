"""Frames (plus an optional waveform) to one playable MP4, through ffmpeg.

The first video model in the engine needs this, but nothing here knows about any model: it takes
decoded frames the way ``file_store`` takes a decoded image. Frames stream in over stdin so a 15
second clip never materialises as one encoded blob in memory.

Lives in ``runtime/`` rather than ``studio/timeline/`` because the take store needs it and runtime
must not import studio. Torch-free: tensors are accepted, but only through ``.detach().cpu()``.
"""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ComponentError
from ..ffmpeg import ffmpeg_exe

#: Chosen for playback rather than fidelity: these takes are served straight into a browser
#: `<video>`, which needs H.264 in yuv420p, and faststart so it plays before the file is buffered.
_VIDEO_CODEC = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18")
_AUDIO_CODEC = ("-c:a", "aac", "-b:a", "192k")


def build_encode_args(
    out_path: str,
    *,
    width: int,
    height: int,
    fps: float,
    audio_path: str | None = None,
) -> list[str]:
    """The ffmpeg arg vector (excluding the binary), with raw frames arriving on stdin."""
    args = [
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:g}",
        "-i", "-",
    ]
    if audio_path is not None:
        args += ["-i", audio_path]
    # yuv420p cannot represent an odd dimension, so pad rather than reject a canvas we were handed.
    if width % 2 or height % 2:
        args += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    args += [*_VIDEO_CODEC]
    if audio_path is not None:
        # -shortest: a soundtrack a few samples longer than the frames must not extend the video.
        args += [*_AUDIO_CODEC, "-shortest"]
    args += ["-movflags", "+faststart", out_path]
    return args


def encode_video_file(
    path: Path,
    frames: Any,
    *,
    fps: float,
    audio: Any = None,
    sample_rate: int | None = None,
) -> None:
    """Write ``frames`` to ``path`` as an MP4, muxing ``audio`` into the same file when given."""
    exe = ffmpeg_exe()
    if exe is None:
        raise ComponentError(
            "Rendering video needs ffmpeg, which was not found. Install the `server` extra "
            "(which bundles it) or put ffmpeg on PATH."
        )
    stream = _rgb_frames(frames)
    try:
        first = next(stream)
    except StopIteration:
        raise ComponentError("The model returned no video frames.") from None
    height, width, _ = first.shape

    audio_path: Path | None = None
    if audio is not None:
        if not sample_rate:
            raise ComponentError("Muxing audio needs its sample rate.")
        audio_path = path.with_suffix(".audio.wav")
        write_wav(audio_path, audio, sample_rate)

    args = build_encode_args(
        str(path), width=width, height=height, fps=fps,
        audio_path=str(audio_path) if audio_path else None,
    )
    try:
        _run(exe, args, first, stream, width, height)
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


def write_wav(path: Path, waveform: Any, sample_rate: int) -> None:
    """Write a mono or stereo waveform as 16-bit PCM. Stdlib only, so no codec to negotiate."""
    samples = _pcm16(waveform)
    channels = samples.shape[1]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(samples.tobytes())


def _run(
    exe: str,
    args: list[str],
    first: np.ndarray[Any, Any],
    rest: Iterator[np.ndarray[Any, Any]],
    width: int,
    height: int,
) -> None:
    proc = subprocess.Popen(  # noqa: S603 - exe is ours, args are built above
        [exe, *args], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert proc.stdin is not None
    try:
        for index, frame in enumerate([first, *rest]):
            if frame.shape[:2] != (height, width):
                raise ComponentError(
                    f"Frame {index} is {frame.shape[1]}x{frame.shape[0]}, but the clip started at "
                    f"{width}x{height}. Every frame has to be the same size."
                )
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass  # ffmpeg died early; its stderr below is the useful message, not this
    finally:
        proc.stdin.close()
    # Not communicate(): it flushes stdin, which we just closed, and raises over the real error.
    err = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-6:]
        raise ComponentError("ffmpeg failed writing the video: " + " / ".join(tail))


def _rgb_frames(frames: Any) -> Iterator[np.ndarray[Any, Any]]:
    """Normalise anything frame-shaped into a stream of contiguous HxWx3 uint8 arrays."""
    for frame in _iter_frames(frames):
        yield _to_rgb(frame)


def _iter_frames(frames: Any) -> Iterable[Any]:
    array = _to_numpy(frames)
    # A single 4D block (T,H,W,C) or (T,C,H,W) is one video, not a batch of images.
    if isinstance(array, np.ndarray) and array.ndim == 4:
        return list(array)
    if isinstance(array, np.ndarray) and array.ndim == 3 and _looks_like_single_image(array):
        return [array]
    return array


def _looks_like_single_image(array: np.ndarray[Any, Any]) -> bool:
    return array.shape[-1] in (1, 3, 4)


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):  # a torch tensor
        return value.detach().to("cpu").numpy()
    return value


def _to_rgb(frame: Any) -> np.ndarray[Any, Any]:
    if hasattr(frame, "convert"):  # a PIL image
        return np.ascontiguousarray(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    array = np.asarray(_to_numpy(frame))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ComponentError(f"A video frame must be 2D or 3D, got shape {array.shape}.")
    # Channels-first (C,H,W) is what a torch pipeline hands back; only 1/3/4 can be a channel axis.
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        scaled = array * 255.0 if float(np.nanmax(array, initial=0.0)) <= 1.0 else array
        array = np.nan_to_num(scaled).clip(0, 255).round().astype(np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] != 3:
        raise ComponentError(f"A video frame needs 1, 3 or 4 channels, got {array.shape[2]}.")
    return np.ascontiguousarray(array)


def _pcm16(waveform: Any) -> np.ndarray[Any, Any]:
    """Normalise a waveform to (samples, channels) int16."""
    array = np.asarray(_to_numpy(waveform))
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ComponentError(f"An audio waveform must be 1D or 2D, got shape {array.shape}.")
    # Channels-first (2, samples) is how a stereo model returns it; samples always outnumber them.
    if array.shape[0] < array.shape[1]:
        array = array.T
    if array.dtype != np.int16:
        array = (np.nan_to_num(array).clip(-1.0, 1.0) * 32767.0).round().astype(np.int16)
    return np.ascontiguousarray(array)
