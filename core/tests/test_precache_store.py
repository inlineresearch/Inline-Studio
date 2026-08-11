"""The on-disk dataset cache: what it reuses, and what it must refuse to reuse.

A stale hit is worse than a miss. Training would carry on against latents that no longer match the
dataset or the settings, with nothing in the log to say so, so most of these tests are about the
cache correctly missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from inline_core.training import precache_store as store  # noqa: E402

SETTINGS = {"resolution": 512, "flip": False, "dropout": False, "clip_frames": 1}


def _dataset(root: Path, captions: dict[str, str], body: bytes = b"pixels") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, caption in captions.items():
        (root / f"{name}.png").write_bytes(body)
        (root / f"{name}.txt").write_text(caption, encoding="utf-8")
    return root


def _items() -> list[dict[str, object]]:
    return [
        {"latent": torch.randn(4, 8), "embed": torch.randn(3, 8),
         "uncond": {"embed": torch.randn(2, 8)}},
        {"latent": torch.randn(4, 8), "embed": torch.randn(5, 8)},
    ]


def test_a_cached_encode_comes_back_intact(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "ds", {"a": "a cat"})
    key = store.fingerprint(str(data), "z-image", SETTINGS)
    items = _items()
    store.save(tmp_path / "cache", key, items, {"embed": torch.zeros(1, 8)}, 3.0)

    hit = store.load(tmp_path / "cache", key)
    assert hit is not None
    back, uncond, shift = hit
    assert shift == 3.0
    assert uncond is not None and uncond["embed"].shape == (1, 8)
    assert len(back) == 2
    for got, want in zip(back, items, strict=True):
        assert torch.equal(got["latent"], want["latent"])
    # The nested dict an arch stashes on an item has to survive the round trip too.
    assert torch.equal(back[0]["uncond"]["embed"], items[0]["uncond"]["embed"])


def test_a_missing_cache_is_a_miss_not_an_error(tmp_path: Path) -> None:
    assert store.load(tmp_path / "cache", "nothing-here") is None


def test_editing_a_caption_misses(tmp_path: Path) -> None:
    """The conditioning is encoded from the caption, so reusing it would train on the old text."""
    data = _dataset(tmp_path / "ds", {"a": "a cat"})
    before = store.fingerprint(str(data), "z-image", SETTINGS)
    (data / "a.txt").write_text("a dog", encoding="utf-8")
    assert store.fingerprint(str(data), "z-image", SETTINGS) != before


def test_changing_an_image_misses(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "ds", {"a": "a cat"})
    before = store.fingerprint(str(data), "z-image", SETTINGS)
    (data / "a.png").write_bytes(b"different pixels")
    assert store.fingerprint(str(data), "z-image", SETTINGS) != before


@pytest.mark.parametrize(
    "change", [{"resolution": 1024}, {"flip": True}, {"dropout": True}, {"clip_frames": 22}]
)
def test_changing_a_setting_that_changes_the_tensors_misses(tmp_path: Path, change) -> None:  # type: ignore[no-untyped-def]
    data = _dataset(tmp_path / "ds", {"a": "a cat"})
    before = store.fingerprint(str(data), "z-image", SETTINGS)
    assert store.fingerprint(str(data), "z-image", {**SETTINGS, **change}) != before


def test_a_different_arch_misses(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "ds", {"a": "a cat"})
    assert store.fingerprint(str(data), "z-image", SETTINGS) != store.fingerprint(
        str(data), "minimax-h3", SETTINGS
    )


def test_the_same_dataset_re_exported_hits(tmp_path: Path) -> None:
    """Every run copies the dataset into its own folder, so paths and mtimes differ between runs
    that hold identical images. Keying on those would mean the cache never hit twice."""
    first = _dataset(tmp_path / "run-1" / "dataset", {"a": "a cat", "b": "a dog"})
    second = _dataset(tmp_path / "run-2" / "dataset", {"a": "a cat", "b": "a dog"})
    assert store.fingerprint(str(first), "z-image", SETTINGS) == store.fingerprint(
        str(second), "z-image", SETTINGS
    )


def test_an_interrupted_write_leaves_nothing_to_trust(tmp_path: Path) -> None:
    """Written to a staging folder and renamed, so a half-written cache is never loadable."""
    root = tmp_path / "cache"
    root.mkdir()
    (root / ".abc123.partial").mkdir()
    (root / ".abc123.partial" / "item-00000.safetensors").write_bytes(b"truncated")
    assert store.load(root, "abc123") is None


def test_a_damaged_cache_falls_back_to_encoding(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    (root / "abc123").mkdir(parents=True)
    (root / "abc123" / "index.json").write_text("{ not json", encoding="utf-8")
    assert store.load(root, "abc123") is None
