"""Dataset precache: flip augmentation and the unconditional embedding caption dropout swaps in.

Both are encoded up front, because by training time the VAE and text encoder have been freed to
make room for the transformer - there is no second chance to encode anything.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from inline_core.training import arch as archs  # noqa: E402
from inline_core.training import dataset as ds  # noqa: E402


class _Vae:
    """Records the pixels it was handed, and returns a latent derived from them."""

    def __init__(self) -> None:
        self.seen: list[Any] = []
        self.config = type("C", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

    def encode(self, pixels: Any) -> Any:
        self.seen.append(pixels.clone())
        latent = pixels[:, :, ::8, ::8]
        dist = type("D", (), {"sample": lambda _self: latent})()
        return type("O", (), {"latent_dist": dist})()


class _Components:
    def __init__(self) -> None:
        self.vae = _Vae()
        self.captions: list[str] = []


def _fake_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    components.captions.append(caption)
    # A tensor that encodes the caption's length, so a swapped-in item is identifiable.
    return {"embed": torch.full((4, 2), float(len(caption)))}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    from PIL import Image

    for i in range(3):
        img = Image.new("RGB", (64, 64), (10 * i, 20, 30))
        # An asymmetric mark, so a mirrored copy is distinguishable from the original.
        img.putpixel((2, 2), (255, 255, 255))
        img.save(tmp_path / f"{i:04d}.png")
        (tmp_path / f"{i:04d}.txt").write_text(f"caption {i}", encoding="utf-8")
    monkeypatch.setattr(ds, "_zimage_caption", _fake_caption)
    return tmp_path


def test_flip_doubles_the_dataset_with_genuinely_mirrored_pixels(dataset) -> None:
    components = _Components()

    plain = ds.precache(str(dataset), components, archs.Z_IMAGE, "cpu", torch.float32, 32)
    seen_plain = len(components.vae.seen)

    components = _Components()
    flipped = ds.precache(
        str(dataset), components, archs.Z_IMAGE, "cpu", torch.float32, 32, flip=True
    )

    assert len(plain) == 3
    assert len(flipped) == 6
    assert len(components.vae.seen) == 2 * seen_plain
    # Every second entry is the mirror of the one before it, encoded from flipped pixels rather
    # than produced by flipping a cached latent.
    original, mirror = components.vae.seen[0], components.vae.seen[1]
    assert not torch.equal(original, mirror)
    assert torch.equal(original, torch.flip(mirror, dims=[-1]))


def test_flip_keeps_each_image_paired_with_its_own_caption(dataset) -> None:
    components = _Components()

    ds.precache(str(dataset), components, archs.Z_IMAGE, "cpu", torch.float32, 32, flip=True)

    assert components.captions == [
        "caption 0", "caption 0", "caption 1", "caption 1", "caption 2", "caption 2",
    ]


def test_the_unconditional_embedding_encodes_an_empty_caption(dataset) -> None:
    components = _Components()

    empty = ds.precache_empty(components, archs.Z_IMAGE, "cpu")

    assert components.captions == [""]
    assert empty["embed"].shape == (4, 2)
    assert float(empty["embed"][0][0]) == 0.0  # len("") == 0, so it is distinguishable


def test_precache_refuses_an_empty_folder(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="empty"):
        ds.precache(str(tmp_path), _Components(), archs.Z_IMAGE, "cpu", torch.float32, 32)
