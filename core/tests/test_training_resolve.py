"""Which weight file a training run picks, when several architectures share the models folder.

The regression this pins: ``vae/`` and ``text_encoders/`` are shared by every architecture, so
"the first file in the folder" hands Z-Image the Qwen3-VL encoder the moment Krea 2 is installed
next to it - and the mismatch only surfaces as a meta-tensor error deep inside the load.
"""

from __future__ import annotations

import pytest

models = pytest.importorskip("inline_core.training.models")
from inline_core.training import arch as archs  # noqa: E402


@pytest.fixture
def both_installed(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A models dir holding Z-Image and Krea 2 side by side, as a real user's would."""
    root = tmp_path / "models"
    files = {
        "diffusion_models": [
            "krea2_raw_bf16.safetensors",
            "krea2_turbo_bf16.safetensors",
            "z_image_bf16.safetensors",
        ],
        "vae": ["ae.safetensors", "qwen_image_vae_diffusers.safetensors"],
        "text_encoders": ["qwen3vl_4b_bf16.safetensors", "qwen_3_4b.safetensors"],
    }
    for category, names in files.items():
        (root / category).mkdir(parents=True)
        for name in names:
            (root / category / name).write_bytes(b"")
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    for var in (
        "INLINE_ZIMAGE_MODEL", "INLINE_KREA2_MODEL",
        "INLINE_ZIMAGE_VAE", "INLINE_KREA2_VAE",
        "INLINE_ZIMAGE_TEXT_ENCODER", "INLINE_KREA2_TEXT_ENCODER",
    ):
        monkeypatch.delenv(var, raising=False)
    return root


def test_zimage_is_not_handed_krea2s_files(both_installed) -> None:
    root = both_installed

    # Alphabetically krea2_raw and qwen3vl_4b both sort first - neither may be picked here.
    assert models._require(root, archs.Z_IMAGE, "diffusion_models").endswith(
        "z_image_bf16.safetensors"
    )
    assert models._require(root, archs.Z_IMAGE, "text_encoders").endswith("qwen_3_4b.safetensors")
    assert models._require(root, archs.Z_IMAGE, "vae").endswith("ae.safetensors")


def test_krea2_is_not_handed_zimages_files(both_installed) -> None:
    root = both_installed

    assert models._require(root, archs.KREA2, "text_encoders").endswith(
        "qwen3vl_4b_bf16.safetensors"
    )
    assert models._require(root, archs.KREA2, "vae").endswith(
        "qwen_image_vae_diffusers.safetensors"
    )


def test_each_krea2_base_mode_picks_its_own_checkpoint(both_installed) -> None:
    root = both_installed

    assert models._base_file(root, archs.KREA2, "raw").endswith("krea2_raw_bf16.safetensors")
    assert models._base_file(root, archs.KREA2, "turbo_adapter").endswith(
        "krea2_turbo_bf16.safetensors"
    )
    assert models._base_file(root, archs.Z_IMAGE, "deturbo").endswith("z_image_bf16.safetensors")


def test_a_missing_component_names_the_arch_and_the_override(monkeypatch, tmp_path) -> None:
    root = tmp_path / "models"
    (root / "vae").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_KREA2_VAE", raising=False)

    with pytest.raises(RuntimeError, match="INLINE_KREA2_VAE"):
        models._require(root, archs.KREA2, "vae")


def test_an_env_override_still_wins(monkeypatch, both_installed) -> None:
    chosen = both_installed / "elsewhere.safetensors"
    chosen.write_bytes(b"")
    monkeypatch.setenv("INLINE_ZIMAGE_TEXT_ENCODER", str(chosen))

    assert models._require(both_installed, archs.Z_IMAGE, "text_encoders") == str(chosen)
