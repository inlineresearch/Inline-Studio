"""Krea 2 model resolution + requirements presence - pure path logic, no torch invocation.

The contract that matters: a user holding **both** checkpoints must get the one their node asks for,
and the shared VAE/text-encoder must never silently resolve to Z-Image's files sitting in the same
folders (which would fail deep inside a 26GB load instead of in the model popup).
"""

from __future__ import annotations

import pytest

reqs = pytest.importorskip("inline_core.models.krea2.requirements")


def _models_root(monkeypatch, tmp_path):
    root = tmp_path / "models"
    for category in ("diffusion_models", "vae", "text_encoders"):
        (root / category).mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    for var in ("INLINE_KREA2_MODEL", "INLINE_KREA2_VAE", "INLINE_KREA2_TEXT_ENCODER"):
        monkeypatch.delenv(var, raising=False)
    return root


def _write(root, category, name):
    path = root / category / name
    path.write_bytes(b"")
    return path


def test_each_variant_resolves_its_own_checkpoint(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    turbo = _write(root, "diffusion_models", "krea2_turbo_bf16.safetensors")
    raw = _write(root, "diffusion_models", "krea2_raw_bf16.safetensors")

    assert reqs.resolve_diffusion("turbo") == turbo
    assert reqs.resolve_diffusion("raw") == raw


def test_the_other_variants_file_is_never_used_as_a_fallback(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", "krea2_raw_bf16.safetensors")

    # Only RAW is present, so the Turbo node reports missing rather than quietly running RAW at
    # 8 CFG-free steps, which would look broken instead of unconfigured.
    assert reqs.resolve_diffusion("turbo") is None
    assert reqs.resolve_diffusion("raw") is not None


def test_a_renamed_krea_checkpoint_still_resolves(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    renamed = _write(root, "diffusion_models", "Krea-2-Turbo-my-copy.safetensors")

    assert reqs.resolve_diffusion("turbo") == renamed


def test_z_image_files_do_not_satisfy_krea2(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", "z_image_bf16.safetensors")
    _write(root, "vae", "ae.safetensors")
    _write(root, "text_encoders", "qwen_3_4b.safetensors")

    assert reqs.resolve_diffusion("turbo") is None
    assert reqs.resolve_vae() is None
    assert reqs.resolve_text_encoder() is None


def test_param_and_env_overrides_win(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", "krea2_turbo_bf16.safetensors")
    picked = _write(root, "diffusion_models", "krea2_turbo_alt.safetensors")
    assert reqs.resolve_diffusion("turbo", {"model": "krea2_turbo_alt.safetensors"}) == picked

    elsewhere = tmp_path / "elsewhere.safetensors"
    elsewhere.write_bytes(b"")
    monkeypatch.setenv("INLINE_KREA2_MODEL", str(elsewhere))
    assert reqs.resolve_diffusion("turbo") == elsewhere


def test_requirements_report_presence_per_component(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    _write(root, "diffusion_models", "krea2_turbo_bf16.safetensors")
    _write(root, "text_encoders", reqs.TEXT_ENCODER_FILE)

    components = {c.id: c for c in reqs.krea2_requirements("turbo")}

    assert [c.id for c in reqs.krea2_requirements("turbo")] == [
        "diffusion", "text_encoder", "vae", "depth_controlnet",
    ]
    assert components["diffusion"].present
    assert components["text_encoder"].present
    assert not components["vae"].present
    # The VAE comes from Qwen-Image, not Comfy-Org: ComfyUI's copy is a layout diffusers can't read.
    assert components["vae"].repo == "Qwen/Qwen-Image"
    assert components["diffusion"].repo == "Comfy-Org/Krea-2"


def test_requirements_name_the_variant_so_the_popup_downloads_the_right_file(monkeypatch, tmp_path):
    _models_root(monkeypatch, tmp_path)

    turbo = {c.id: c for c in reqs.krea2_requirements("turbo")}["diffusion"]
    raw = {c.id: c for c in reqs.krea2_requirements("raw")}["diffusion"]

    assert turbo.filename == "krea2_turbo_bf16.safetensors"
    assert raw.filename == "krea2_raw_bf16.safetensors"
    assert turbo.repo_file == "diffusion_models/krea2_turbo_bf16.safetensors"
    assert turbo.local_path == "diffusion_models/krea2_turbo_bf16.safetensors"


def test_footprint_sizes_the_files_that_exist(monkeypatch, tmp_path):
    root = _models_root(monkeypatch, tmp_path)
    model = root / "diffusion_models" / "krea2_turbo_bf16.safetensors"
    model.write_bytes(b"x" * 2048)

    sizes = reqs.footprint_bytes(model, None, "")

    assert sizes == {
        "diffusion_bytes": 2048,
        "text_encoder_bytes": 0,
        "vae_bytes": 0,
        "controlnet_bytes": 0,
    }


def test_depth_control_is_opt_in_and_never_auto_resolved(monkeypatch, tmp_path):
    """A wired-but-unpicked depth map falls to ``auto_depth_control``; resolve stays explicit."""
    root = _models_root(monkeypatch, tmp_path)
    (root / "controlnet").mkdir()
    monkeypatch.delenv("INLINE_KREA2_CONTROL", raising=False)

    lora = root / "controlnet" / reqs.DEPTH_CONTROL_FILE
    lora.write_bytes(b"")
    # Present on disk, but resolve won't pick it without an explicit dropdown choice.
    assert reqs.resolve_depth_control() is None
    assert reqs.resolve_depth_control({"depth_controlnet": reqs.DEPTH_CONTROL_FILE}) == lora
    # auto is what the runner consults once a control map is actually wired.
    assert reqs.auto_depth_control() == lora
    assert reqs.depth_control_present() is True


def test_auto_depth_control_ignores_a_z_image_controlnet(monkeypatch, tmp_path):
    """The Z-Image ControlNet shares ``controlnet/``; auto must not grab it for a Krea 2 node."""
    root = _models_root(monkeypatch, tmp_path)
    (root / "controlnet").mkdir()
    (root / "controlnet" / "Z-Image-Turbo-Fun-Controlnet-Union.safetensors").write_bytes(b"")

    assert reqs.auto_depth_control() is None
    assert reqs.depth_control_present() is False
