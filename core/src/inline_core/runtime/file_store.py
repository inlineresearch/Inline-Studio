"""A file-backed take store: writes a decoded image to <root>/<take_id>.png and records the take."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        data = path.read_bytes()
        return Take(
            id=take_id,
            run_id=run_id,
            node_id=node_id,
            kind=MediaKind.IMAGE,
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
