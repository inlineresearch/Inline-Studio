from __future__ import annotations

from pathlib import Path

import numpy as np
from inline_core.media import MediaKind
from inline_core.runtime.file_store import FileTakeStore


def test_file_store_writes_png(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255

    take = store.save("run1", "n1", image, {"seed": 7})

    assert take.kind is MediaKind.IMAGE
    assert take.hash.startswith("sha256-")
    assert take.params["seed"] == 7
    written = Path(take.uri)
    assert written.exists() and written.suffix == ".png" and written.stat().st_size > 0


def test_file_store_is_content_addressed_hash(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)
    image = np.full((4, 4, 3), 128, dtype=np.uint8)

    a = store.save("run1", "n1", image, {})
    b = store.save("run2", "n2", image, {})

    assert a.hash == b.hash  # same pixels -> same content hash
    assert a.id != b.id
