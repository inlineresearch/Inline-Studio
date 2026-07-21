"""A file-backed take store: an image to <root>/<take_id>.png, a video to <root>/<take_id>.mp4."""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..errors import ComponentError
from ..ffmpeg import ffmpeg_exe
from ..media import MediaKind
from ..takes import Take
from .store import TakeStore


class FileTakeStore(TakeStore):
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, run_id: str, node_id: str, image: Any, params: dict[str, Any]) -> Take:
        self._root.mkdir(parents=True, exist_ok=True)
        take_id = f"take_{uuid4().hex[:12]}"
        path = self._root / f"{take_id}.png"
        _to_pil(image).save(path, format="PNG")
        return self._take(take_id, run_id, node_id, MediaKind.IMAGE, path, params)

    def save_video(
        self,
        run_id: str,
        node_id: str,
        frames: Any,
        params: dict[str, Any],
        fps: float = 16.0,
    ) -> Take:
        self._root.mkdir(parents=True, exist_ok=True)
        take_id = f"take_{uuid4().hex[:12]}"
        path = self._root / f"{take_id}.mp4"
        _encode_mp4(_to_frames(frames), path, fps)
        return self._take(take_id, run_id, node_id, MediaKind.VIDEO, path, params)

    def _take(
        self,
        take_id: str,
        run_id: str,
        node_id: str,
        kind: MediaKind,
        path: Path,
        params: dict[str, Any],
    ) -> Take:
        data = path.read_bytes()
        return Take(
            id=take_id,
            run_id=run_id,
            node_id=node_id,
            kind=kind,
            uri=str(path),
            hash=f"sha256-{hashlib.sha256(data).hexdigest()}",
            params=dict(params),
            created_at=int(time.time() * 1000),
        )


def _to_frames(frames: Any) -> list[Any]:
    """Normalise to a list of uint8 HxWx3 arrays. Accepts a sequence, or a stacked (T,H,W,C)
    array/tensor - a torch video latent decode hands back the latter."""
    import numpy as np

    if hasattr(frames, "detach"):  # a torch tensor
        frames = frames.detach().to("cpu").numpy()
    array = np.asarray(frames) if not isinstance(frames, list | tuple) else None
    seq = list(array) if array is not None and array.ndim == 4 else list(frames)
    if not seq:
        raise ComponentError("Cannot save a video take with no frames.")
    return [np.asarray(_to_pil(f).convert("RGB")) for f in seq]


def _encode_mp4(frames: list[Any], path: Path, fps: float) -> None:
    """Pipe raw RGB into ffmpeg. yuv420p + even dimensions so the result plays everywhere."""
    exe = ffmpeg_exe()
    if exe is None:
        raise ComponentError(
            "ffmpeg is required to save a video take but was not found. "
            "Install ffmpeg, or `pip install imageio-ffmpeg`."
        )
    height, width = frames[0].shape[:2]
    args = [
        exe, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(path),
    ]
    proc = subprocess.run(
        args,
        input=b"".join(f.tobytes() for f in frames),
        capture_output=True,
    )
    if proc.returncode != 0 or not path.exists():
        tail = proc.stderr.decode(errors="ignore").strip().splitlines()[-3:]
        raise ComponentError("ffmpeg failed to encode the video take: " + " ".join(tail))


def _to_pil(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if hasattr(image, "detach"):  # a torch tensor
        image = image.detach().to("cpu").numpy()
    import numpy as np

    array = np.asarray(image)
    if array.dtype != np.uint8:
        scaled = array * 255.0 if float(array.max(initial=0.0)) <= 1.0 else array
        array = scaled.clip(0, 255).round().astype(np.uint8)
    return Image.fromarray(array)
