"""The model-agnostic loader core (models/loaders.py) - the torch-free surface: the arch registry,
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
    # The bundled assets are configs + tokenizer only - never the multi-GB weights.
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
    """A ``.complete`` marker means the one-time fetch already ran - ``ensure_assets`` returns
    before importing huggingface_hub or touching the network, so generation stays fully offline."""
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path))
    root = loaders.assets_root("z-image")
    root.mkdir(parents=True)
    (root / ".complete").write_text("ok")
    # No network/hub install needed: the marker short-circuits at the top of ensure_assets.
    assert loaders.ensure_assets("z-image") == root


def test_device_key_maps_none_to_cpu():
    assert loaders._device_key(None) == "cpu"
    assert loaders._device_key("cuda:0") == "cuda:0"


def _prime_assets(monkeypatch, tmp_path):
    """A ready assets bundle (marker + a text_encoder config) so ensure_assets stays offline."""
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path))
    root = loaders.assets_root("z-image")
    (root / "text_encoder").mkdir(parents=True)
    (root / "text_encoder" / "config.json").write_text("{}")
    (root / ".complete").write_text("ok")
    return root


def test_staged_encoder_dir_places_config_next_to_weights(monkeypatch, tmp_path):
    """Encoder loads from a staging dir (config + weights linked as model.safetensors) so
    transformers uses its mmap->device streaming loader instead of a full-RAM state_dict."""
    _prime_assets(monkeypatch, tmp_path)
    weights = tmp_path / "qwen_3_4b.safetensors"
    weights.write_bytes(b"weights")
    stage = loaders._staged_encoder_dir("z-image", str(weights))
    assert (stage / "model.safetensors").read_bytes() == b"weights"
    assert (stage / "config.json").is_file()
    assert (stage / ".complete").is_file()
    # Idempotent: a second call returns the same staged dir without re-linking.
    assert loaders._staged_encoder_dir("z-image", str(weights)) == stage


def test_unload_components_evicts_all_but_kept_files():
    loaders._CACHE.clear()
    keys = {
        ("z-image", "diffusion", "/models/old.safetensors", "fp16", "none", "cuda:0"): object(),
        ("z-image", "vae", "/models/ae.safetensors", "fp32", "none", "cuda:0"): object(),
        ("z-image", "text_encoder", "/models/qwen.safetensors", "fp16", "int8", "cuda:0"): object(),
    }
    loaders._CACHE.update(keys)
    loaders.unload_components(keep_files={"/models/ae.safetensors", "/models/qwen.safetensors"})
    remaining = {k[2] for k in loaders._CACHE}
    assert remaining == {"/models/ae.safetensors", "/models/qwen.safetensors"}
    loaders._CACHE.clear()
