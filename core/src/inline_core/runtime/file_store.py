"""A file-backed take store: writes a decoded output under <root>/<take_id>.<ext> and records it."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..media import MediaKind
from ..takes import Take
from .store import TakeStore
from .video_encode import encode_video_file, write_wav


class FileTakeStore(TakeStore):
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, run_id: str, node_id: str, image: Any, params: dict[str, Any]) -> Take:
        path = self._path("png")
        _to_pil(image).save(path, format="PNG")
        return self._take(run_id, node_id, MediaKind.IMAGE, path, params)

    def save_video(
        self,
        run_id: str,
        node_id: str,
        frames: Any,
        params: dict[str, Any],
        *,
        fps: float,
        audio: Any = None,
        sample_rate: int | None = None,
    ) -> Take:
        path = self._path("mp4")
        encode_video_file(path, frames, fps=fps, audio=audio, sample_rate=sample_rate)
        return self._take(run_id, node_id, MediaKind.VIDEO, path, params)

    def save_audio(
        self, run_id: str, node_id: str, waveform: Any, params: dict[str, Any], *, sample_rate: int
    ) -> Take:
        path = self._path("wav")
        write_wav(path, waveform, sample_rate)
        return self._take(run_id, node_id, MediaKind.AUDIO, path, params)

    def _path(self, ext: str) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root / f"take_{uuid4().hex[:12]}.{ext}"

    def _take(
        self, run_id: str, node_id: str, kind: MediaKind, path: Path, params: dict[str, Any]
    ) -> Take:
        data = path.read_bytes()
        return Take(
            id=path.stem,
            run_id=run_id,
            node_id=node_id,
            kind=kind,
            uri=str(path),
            hash=f"sha256-{hashlib.sha256(data).hexdigest()}",
            params=dict(params),
            created_at=int(time.time() * 1000),
        )


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
