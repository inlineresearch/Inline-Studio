"""Z-Image model resolution + requirements presence — pure path logic, no torch/model invocation.

Imports the requirements module (which lives in the zimage package, so torch/diffusers must be
present); skipped cleanly if the zimage extra is absent. Key contract for Phase 3: **no repo-id
fallback** — a missing model resolves to ``None`` (reported missing), never a silent download.
"""

from __future__ import annotations

import pytest

reqs = pytest.importorskip("inline_core.models.zimage.requirements")


def _models_root(monkeypatch, tmp_path):
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_ZIMAGE_MODEL", raising=False)
    monkeypatch.delenv("INLINE_ZIMAGE_VAE", raising=False)
    monkeypatch.delenv("INLINE_ZIMAGE_TEXT_ENCODER", raising=False)
    return root


def test_single_file_dropped_in_diffusion_models(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    f = root / "diffusion_models" / "z_image_turbo_bf16.safetensors"
    f.write_bytes(b"")
    assert reqs.resolve_diffusion() == ("single_file", str(f))


def test_prefers_z_image_named_file_over_other_weights(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    (root / "diffusion_models" / "some_other_model.safetensors").write_bytes(b"")
    zi = root / "diffusion_models" / "z-image-turbo.safetensors"
    zi.write_bytes(b"")
    assert reqs.resolve_diffusion() == ("single_file", str(zi))


def test_param_override_wins(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    (root / "diffusion_models" / "z_image_turbo_bf16.safetensors").write_bytes(b"")
    pick = root / "diffusion_models" / "z_image_alt.safetensors"
    pick.write_bytes(b"")
    got = reqs.resolve_diffusion({"model": "z_image_alt.safetensors"})
    assert got == ("single_file", str(pick))


def test_env_file_is_single_file(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    f = tmp_path / "elsewhere.safetensors"
    f.write_bytes(b"")
    monkeypatch.setenv("INLINE_ZIMAGE_MODEL", str(f))
    assert reqs.resolve_diffusion() == ("single_file", str(f))


def test_env_repo_id_is_pipeline(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    monkeypatch.setenv("INLINE_ZIMAGE_MODEL", "some-org/some-repo")
    assert reqs.resolve_diffusion() == ("pipeline", "some-org/some-repo")


def test_diffusers_dir_is_pipeline(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    d = root / "diffusion_models" / "Z-Image-Turbo"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    assert reqs.resolve_diffusion() == ("pipeline", str(d))


def test_no_repo_fallback_when_nothing_local(monkeypatch, tmp_path):
    """The Phase 3 rule: nothing local -> None (missing), never a repo-id auto-download."""
    _models_root(monkeypatch, tmp_path)
    assert reqs.resolve_diffusion() is None


# --- requirements presence (the popup's data) ---------------------------------------------------


def test_requirements_all_missing_on_empty_root(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    components = reqs.zimage_requirements()
    assert {c.id for c in components} == {"diffusion", "vae", "text_encoder"}
    assert all(not c.present for c in components)
    assert all(c.repo == reqs.BASE_REPO for c in components)


def test_pipeline_dir_satisfies_vae_and_text_encoder(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    d = root / "diffusion_models" / "Z-Image-Turbo"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    by_id = {c.id: c for c in reqs.zimage_requirements()}
    # A whole-pipeline diffusers folder contains the VAE + text-encoder, so all three read present.
    assert by_id["diffusion"].present
    assert by_id["vae"].present
    assert by_id["text_encoder"].present


def test_single_file_needs_local_vae_and_text_encoder(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    (root / "diffusion_models" / "z-image-turbo.safetensors").write_bytes(b"")
    by_id = {c.id: c for c in reqs.zimage_requirements()}
    assert by_id["diffusion"].present  # the single file is there
    assert not by_id["vae"].present  # but VAE + text-encoder are not
    assert not by_id["text_encoder"].present

    # Split-file loading: a single weight file per category (a bare config dir no longer counts).
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "ae.safetensors").write_bytes(b"")
    (root / "text_encoders").mkdir(parents=True)
    (root / "text_encoders" / "qwen_3_4b.safetensors").write_bytes(b"")
    by_id = {c.id: c for c in reqs.zimage_requirements()}
    assert by_id["vae"].present and by_id["text_encoder"].present


def test_config_only_dir_is_not_present(monkeypatch, tmp_path):
    """A folder with only a config (no weights) is not a loadable single file — reads as missing."""
    root = _models_root(monkeypatch, tmp_path)
    (root / "vae" / "z-image-turbo").mkdir(parents=True)
    (root / "vae" / "z-image-turbo" / "config.json").write_text("{}")
    by_id = {c.id: c for c in reqs.zimage_requirements()}
    assert not by_id["vae"].present


def test_resolvers_prefer_dropdown_pick_then_exact_file(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "ae.safetensors").write_bytes(b"")
    alt = root / "vae" / "my_vae.safetensors"
    alt.write_bytes(b"")
    # No pick -> the recommended split file; an explicit dropdown pick wins.
    assert reqs.resolve_vae() == root / "vae" / "ae.safetensors"
    assert reqs.resolve_vae({"vae": "my_vae.safetensors"}) == alt


def test_component_points_at_split_files(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)
    by_id = {c.id: c for c in reqs.zimage_requirements()}
    assert by_id["diffusion"].repo == "Comfy-Org/z_image"
    assert by_id["diffusion"].repo_file == "split_files/diffusion_models/z_image_bf16.safetensors"
    assert by_id["vae"].repo_file == "split_files/vae/ae.safetensors"
    assert by_id["text_encoder"].repo_file == "split_files/text_encoders/qwen_3_4b.safetensors"
    # The file lands flat in its category so the node's dropdown lists it.
    assert by_id["vae"].local_path == "vae/ae.safetensors"


def test_download_target_is_the_category_dir(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    vae = next(c for c in reqs.zimage_requirements() if c.id == "vae")
    target = reqs.download_target(vae)
    # Downloads land flat under models/<category>/ (never a hidden HF cache).
    assert str(target).startswith(str(root))
    assert target == root / "vae"
