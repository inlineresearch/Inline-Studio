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
    paths = [asset.path for asset in spec.asset_files]
    # The bundled assets are configs + tokenizer only - never the multi-GB weights.
    assert "transformer/config.json" in paths
    assert "tokenizer/tokenizer.json" in paths
    assert not any(name.endswith(".safetensors") for name in paths)


def test_krea2_spec_sources_its_assets_from_the_ungated_qwen_repos():
    spec = loaders._spec("krea2")
    targets = [asset.target(spec) for asset in spec.asset_files]

    # Krea 2's own repos are gated, so nothing here may point at them.
    assert not any(repo.startswith("krea/") for repo, _ in targets)
    assert ("Qwen/Qwen-Image", "vae/config.json") in targets
    assert ("Qwen/Qwen3-VL-4B-Instruct", "text_encoder/config.json") in targets
    assert not any(local.endswith(".safetensors") for _, local in targets)


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


def test_unload_components_evicts_the_same_file_at_another_quantization():
    """Adding a ControlNet flips the fit plan from resident to int8, so the same transformer file
    is cached twice. Without a quant-aware sweep both stay resident - two full-size models."""
    loaders._CACHE.clear()
    int8 = ("z-image", "diffusion", "/m.safetensors", "fp16", "int8", "cuda:0")
    plain = ("z-image", "diffusion", "/m.safetensors", "fp16", "none", "cuda:0")
    encoder = ("z-image", "text_encoder", "/qwen.safetensors", "fp16", "int8", "cuda:0")
    loaders._CACHE.update({int8: object(), plain: object(), encoder: object()})

    loaders.unload_components(
        keep_files={"/m.safetensors", "/qwen.safetensors"}, keep_quant="none"
    )

    assert plain in loaders._CACHE
    assert int8 not in loaders._CACHE
    assert encoder not in loaders._CACHE  # the encoder quantizes too
    loaders._CACHE.clear()


def test_unload_components_keeps_never_quantized_kinds_across_a_quant_switch():
    """The VAE and the ControlNet always key as NONE whatever the plan, so a quant-aware sweep must
    not read their slot as a mismatch and drop them on every control run."""
    loaders._CACHE.clear()
    vae = ("z-image", "vae", "/ae.safetensors", "fp32", "none", "cuda:0")
    controlnet = ("z-image", "controlnet", "/union.safetensors", "fp16", "none", "cuda:0")
    loaders._CACHE.update({vae: object(), controlnet: object()})

    loaders.unload_components(
        keep_files={"/ae.safetensors", "/union.safetensors"}, keep_quant="int8"
    )

    assert vae in loaders._CACHE and controlnet in loaders._CACHE
    loaders._CACHE.clear()
