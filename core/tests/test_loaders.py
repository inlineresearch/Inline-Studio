"""The model-agnostic loader core (models/loaders.py) — the torch-free surface: the arch registry,
asset paths, and the fetch-once ``.complete`` short-circuit. The heavy diffusers/transformers loads
import inside functions on purpose, so this module imports without the ``zimage`` extra; the actual
weight loading needs a GPU and real files and is exercised by running the app, not here."""

from __future__ import annotations

import pytest

from inline_core.errors import ComponentError
from inline_core.models import loaders


def test_zimage_spec_is_registered():
    spec = loaders._spec("z-image")
    assert spec.assets_repo == "Tongyi-MAI/Z-Image-Turbo"
    # The bundled assets are configs + tokenizer only — never the multi-GB weights.
    assert "transformer/config.json" in spec.asset_files
    assert "tokenizer/tokenizer.json" in spec.asset_files
    assert not any(name.endswith(".safetensors") for name in spec.asset_files)


def test_unknown_arch_raises():
    with pytest.raises(ComponentError):
        loaders._spec("no-such-arch")


def test_assets_root_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path))
    assert loaders.assets_root("z-image") == tmp_path / "assets" / "z-image"


def test_ensure_assets_short_circuits_on_complete_marker(monkeypatch, tmp_path):
    """A ``.complete`` marker means the one-time fetch already ran — ``ensure_assets`` returns
    before importing huggingface_hub or touching the network, so generation stays fully offline."""
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path))
    root = loaders.assets_root("z-image")
    root.mkdir(parents=True)
    (root / ".complete").write_text("ok")
    # No network/hub install needed: the marker short-circuits at the top of ensure_assets.
    assert loaders.ensure_assets("z-image") == root
