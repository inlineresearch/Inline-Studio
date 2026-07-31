"""Diffusers-format checkpoint folders: the only practical way to put dev on a 24 GB card.

A prequantized dev build ships as a folder of NF4 shards plus a config, rather than one
consolidated file. Three things follow, and all three are load-bearing:

- geometry comes from ``config.json`` instead of a derived tensor header;
- the weights must **not** be quantized again (their on-disk size already is their resident size);
- the encoder and the transformer cannot be co-resident, so the prompt is encoded first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inline_core.models.flux2 import requirements as reqs
from inline_core.models.flux2 import variants as V

DEV_CONFIG = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.36.0.dev0",
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 32, 32, 32],
    "eps": 1e-06,
    "in_channels": 128,
    "joint_attention_dim": 15360,
    "mlp_ratio": 3.0,
    "num_attention_heads": 48,
    "num_layers": 8,
    "num_single_layers": 48,
    "out_channels": None,
    "patch_size": 1,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}
#: Verbatim from diffusers/FLUX.2-dev-bnb-4bit.
NF4 = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}


def _folder(path: Path, config: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"0" * 2048)
    return path


@pytest.fixture
def models(tmp_path: Path, monkeypatch: Any) -> Path:
    root = tmp_path / "models"
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    for category in ("diffusion_models", "vae", "text_encoders"):
        (root / category).mkdir(parents=True)
    reqs._IDENTIFIED.clear()
    return root


def test_a_folder_states_its_geometry_instead_of_deriving_it(models: Path) -> None:
    folder = _folder(models / "diffusion_models" / "flux2-dev-nf4", DEV_CONFIG)
    config = V.config_for(folder)
    assert config is not None
    assert config["num_layers"] == 8 and config["num_single_layers"] == 48
    assert config["joint_attention_dim"] == 15360
    assert not any(k.startswith("_") for k in config), "diffusers bookkeeping is stripped"
    assert V.detect(folder) is V.get("dev")


def test_a_folder_from_another_architecture_is_left_alone(models: Path) -> None:
    other = _folder(models / "diffusion_models" / "something-else", {"_class_name": "UNet2DModel"})
    assert V.config_for(other) is None
    assert V.detect(other) is None
    assert reqs.resolve_diffusion() is None


def test_a_prequantized_folder_is_flagged_so_it_is_not_quantized_again(models: Path) -> None:
    plain = _folder(models / "diffusion_models" / "a-plain", DEV_CONFIG)
    quantized = _folder(
        models / "diffusion_models" / "b-nf4", {**DEV_CONFIG, "quantization_config": NF4}
    )
    assert V.is_prequantized(plain) is False
    assert V.is_prequantized(quantized) is True
    # A single file never carries its own quantization config.
    assert V.is_prequantized(models / "diffusion_models" / "nope.safetensors") is False


def test_folders_are_resolvable_and_pickable(models: Path) -> None:
    folder = _folder(models / "diffusion_models" / "flux2-dev-nf4", DEV_CONFIG)
    assert reqs.resolve_diffusion() == folder
    # An explicit dropdown pick names the folder; this used to fail an is_file() check.
    assert reqs.resolve_diffusion({"model": "flux2-dev-nf4"}) == folder
    assert reqs.resolved_variant({"model": "flux2-dev-nf4"}) is V.get("dev")


def test_a_folder_encoder_is_matched_by_its_text_width(models: Path) -> None:
    # Mistral-3 is legitimately multimodal, so unlike the single-file check a vision tower must not
    # disqualify it - the text stack's width is what has to line up (5120 * 3 == 15360).
    encoder = models / "text_encoders" / "flux2-dev-mistral-nf4"
    encoder.mkdir(parents=True)
    (encoder / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Mistral3ForConditionalGeneration"],
                "model_type": "mistral3",
                "text_config": {"hidden_size": 5120, "num_hidden_layers": 40},
                "quantization_config": NF4,
            }
        )
    )
    _folder(models / "diffusion_models" / "flux2-dev-nf4", DEV_CONFIG)
    assert reqs._encoder_width(encoder) == 5120
    assert reqs.resolve_text_encoder() == encoder


def test_a_folders_footprint_is_the_sum_of_its_files(models: Path) -> None:
    folder = _folder(models / "diffusion_models" / "flux2-dev-nf4", DEV_CONFIG)
    sizes = reqs.footprint_bytes(folder, None, None)
    assert sizes["diffusion_bytes"] > 2048, "config plus shards, not just one file"


def test_staging_engages_only_when_encoder_and_transformer_cannot_coexist() -> None:
    """The decision that keeps dev from OOMing during the load, before anything can be freed."""
    from inline_core.models.flux2.runner import _needs_staged_encode

    class Policy:
        def __init__(self, mb: int | None) -> None:
            self._mb = mb

        def vram_budget_mb(self) -> int | None:
            return self._mb

    sizes = {"diffusion_bytes": 18_100_000_000, "vae_bytes": 336_000_000,
             "text_encoder_bytes": 15_400_000_000}

    def fake(diffusion=None, vae=None, text_encoder=None, controlnet=None):
        return sizes

    import inline_core.models.flux2.runner as runner_mod

    original = runner_mod.reqs.footprint_bytes
    runner_mod.reqs.footprint_bytes = fake
    try:
        # 24 GB: 33.8 GB together will not fit, 18.4 GB without the encoder will.
        assert _needs_staged_encode("d", "v", "t", Policy(24229)) is True
        # 80 GB: everything is resident, so there is nothing to stage around.
        assert _needs_staged_encode("d", "v", "t", Policy(80 * 1024)) is False
        # 8 GB: even without the encoder it does not fit, so staging would not rescue it.
        assert _needs_staged_encode("d", "v", "t", Policy(8 * 1024)) is False
        # A CPU device reports no budget and takes the normal path.
        assert _needs_staged_encode("d", "v", "t", Policy(None)) is False
    finally:
        runner_mod.reqs.footprint_bytes = original
