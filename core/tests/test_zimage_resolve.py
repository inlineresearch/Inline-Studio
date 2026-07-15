"""Z-Image source resolution: single-file vs diffusers-dir vs repo, and the override priority.

Pure path logic — no torch/model invocation. Imports the runner (torch/diffusers present in the
zimage extra); skipped cleanly if that extra is absent.
"""

from __future__ import annotations

import pytest

runner = pytest.importorskip("inline_core.models.zimage.runner")

_resolve_model = runner._resolve_model
_find_weight_file = runner._find_weight_file
_BASE_REPO = runner._BASE_REPO


def _models_root(monkeypatch, tmp_path):
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_ZIMAGE_MODEL", raising=False)
    return root


def test_single_file_dropped_in_diffusion_models(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    f = root / "diffusion_models" / "z_image_turbo_bf16.safetensors"
    f.write_bytes(b"")
    source, single = _resolve_model()
    assert single is True
    assert source == str(f)


def test_prefers_z_image_named_file_over_other_weights(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    (root / "diffusion_models" / "some_other_model.safetensors").write_bytes(b"")
    zi = root / "diffusion_models" / "z-image-turbo.safetensors"
    zi.write_bytes(b"")
    source, single = _resolve_model()
    assert single is True
    assert source == str(zi)


def test_param_override_wins(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    (root / "diffusion_models" / "z_image_turbo_bf16.safetensors").write_bytes(b"")
    pick = root / "diffusion_models" / "z_image_alt.safetensors"
    pick.write_bytes(b"")
    source, single = _resolve_model({"model": "z_image_alt.safetensors"})
    assert single is True
    assert source == str(pick)


def test_env_file_is_single_file(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    f = tmp_path / "elsewhere.safetensors"
    f.write_bytes(b"")
    monkeypatch.setenv("INLINE_ZIMAGE_MODEL", str(f))
    source, single = _resolve_model()
    assert (source, single) == (str(f), True)


def test_env_repo_id_is_not_single_file(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    monkeypatch.setenv("INLINE_ZIMAGE_MODEL", "some-org/some-repo")
    source, single = _resolve_model()
    assert (source, single) == ("some-org/some-repo", False)


def test_diffusers_dir_is_not_single_file(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    d = root / "diffusion_models" / "Z-Image-Turbo"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    source, single = _resolve_model()
    assert single is False
    assert source == str(d)


def test_falls_back_to_reference_repo(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    source, single = _resolve_model()
    assert (source, single) == (_BASE_REPO, False)
