from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inline_core.ffmpeg import ffmpeg_available
from inline_core.media import MediaKind
from inline_core.runtime.file_store import FileTakeStore
from inline_core.runtime.store import TakeStore


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


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_file_store_writes_mp4(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)
    frames = [np.full((16, 16, 3), i * 20, dtype=np.uint8) for i in range(6)]

    take = store.save_video("run1", "n1", frames, {"steps": 6}, fps=8.0)

    assert take.kind is MediaKind.VIDEO
    assert take.hash.startswith("sha256-")
    assert take.params["steps"] == 6
    written = Path(take.uri)
    assert written.exists() and written.suffix == ".mp4" and written.stat().st_size > 0


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_file_store_accepts_a_stacked_frame_array(tmp_path: Path) -> None:
    store = FileTakeStore(tmp_path)
    stacked = np.zeros((4, 16, 16, 3), dtype=np.uint8)  # (T,H,W,C), what a VAE decode returns

    take = store.save_video("run1", "n1", stacked, {})

    assert take.kind is MediaKind.VIDEO
    assert Path(take.uri).stat().st_size > 0


def test_video_takes_are_opt_in_for_other_stores() -> None:
    """An image-only store stays valid: save_video is concrete and raises, never abstract."""

    class ImageOnly(TakeStore):
        def save(self, run_id, node_id, image, params):  # type: ignore[no-untyped-def]
            raise AssertionError("not called")

    with pytest.raises(NotImplementedError):
        ImageOnly().save_video("r", "n", [], {})
