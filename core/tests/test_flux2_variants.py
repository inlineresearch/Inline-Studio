"""Identifying a FLUX.2 checkpoint from its tensor shapes alone.

These are the tests that let one node serve the whole family: if the derivation drifts, a checkpoint
loads with the wrong geometry and produces noise rather than an error, so the round-trip is pinned
against real ``Flux2Transformer2DModel`` shapes for every width in the family.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from inline_core.models.flux2 import variants as V

#: The verified klein 4B geometry, from black-forest-labs/FLUX.2-klein-4B transformer/config.json.
KLEIN_4B = {
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 32, 32, 32],
    "eps": 1e-06,
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 7680,
    "mlp_ratio": 3.0,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 20,
    "out_channels": None,
    "patch_size": 1,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}
KLEIN_9B = {**KLEIN_4B, "joint_attention_dim": 12288, "num_attention_heads": 32, "num_layers": 6,
            "num_single_layers": 30}
DEV = {**KLEIN_4B, "joint_attention_dim": 15360, "num_attention_heads": 48, "num_layers": 8,
       "num_single_layers": 48, "guidance_embeds": True}


def _shapes(config: dict[str, object]) -> dict[str, list[int]]:
    """Real parameter shapes for a config, straight from diffusers on the meta device."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    with torch.device("meta"):
        model = diffusers.Flux2Transformer2DModel(**config)
    shapes = {name: list(p.shape) for name, p in model.named_parameters()}
    shapes.update({name: list(b.shape) for name, b in model.named_buffers()})
    return shapes


@pytest.mark.parametrize(("label", "config"), [("4b", KLEIN_4B), ("9b", KLEIN_9B), ("dev", DEV)])
def test_config_round_trips_from_tensor_shapes(label: str, config: dict[str, object]) -> None:
    assert V.derive_transformer_config(_shapes(config)) == config


def test_comfy_style_key_prefixes_are_stripped() -> None:
    shapes = {f"model.diffusion_model.{k}": v for k, v in _shapes(KLEIN_4B).items()}
    assert V.derive_transformer_config(shapes) == KLEIN_4B


def test_a_non_flux2_checkpoint_is_not_claimed() -> None:
    assert V.derive_transformer_config({"some.other.weight": [16, 16]}) is None
    assert V.derive_transformer_config({}) is None


@pytest.mark.parametrize(
    ("filename", "config", "expected"),
    [
        ("flux-2-klein-4b.safetensors", KLEIN_4B, "klein-4b"),
        ("flux-2-klein-base-4b.safetensors", KLEIN_4B, "klein-4b-base"),
        ("flux-2-klein-9b.safetensors", KLEIN_9B, "klein-9b"),
        ("flux-2-klein-base-9b.safetensors", KLEIN_9B, "klein-9b-base"),
        ("flux-2-klein-9b-kv.safetensors", KLEIN_9B, "klein-9b-kv"),
        ("flux2_dev_fp8mixed.safetensors", DEV, "dev"),
        # dev has no base or KV split, so those name markers must not steer it elsewhere.
        ("flux2-dev-base-kv.safetensors", DEV, "dev"),
    ],
)
def test_detect_combines_shape_and_name(filename: str, config: dict, expected: str) -> None:
    assert V.detect(filename, _shapes(config)) is V.get(expected)


def test_distilled_and_base_differ_only_by_name() -> None:
    # Architecturally identical, so a mislabeled file is the one case detection cannot save us from
    # and the reason the variant dropdown offers explicit choices.
    shapes = _shapes(KLEIN_4B)
    assert V.detect("a.safetensors", shapes) is V.get("klein-4b")
    assert V.detect("a-base.safetensors", shapes) is V.get("klein-4b-base")


def test_sampler_defaults_follow_distillation() -> None:
    distilled, base = V.get("klein-4b"), V.get("klein-4b-base")
    assert (distilled.steps, distilled.guidance) == (4, 1.0)
    assert (base.steps, base.guidance) == (50, 4.0)
    # Only an undistilled klein runs real CFG; dev is guidance-distilled with no negative path.
    assert base.supports_negative_prompt is True
    assert distilled.supports_negative_prompt is False
    assert V.get("dev").supports_negative_prompt is False


def test_every_variant_maps_to_a_loader_arch_and_pipeline() -> None:
    from inline_core.models import loaders

    for variant in V.VARIANTS:
        assert variant.arch in loaders.SPECS, variant.key
        assert variant.pipeline in ("klein", "klein-kv", "dev"), variant.key
        # The joint width is exactly three encoder layers concatenated, which is what the text
        # encoder is matched on when resolving files.
        assert variant.joint_attention_dim % 3 == 0, variant.key


def test_detect_reads_a_real_safetensors_header(tmp_path: Path) -> None:
    """The on-disk path: a header-only file (no tensor data) is enough to identify a checkpoint."""
    header = {
        name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in _shapes(KLEIN_4B).items()
    }
    blob = json.dumps(header).encode()
    file = tmp_path / "flux-2-klein-base-4b.safetensors"
    file.write_bytes(struct.pack("<Q", len(blob)) + blob)
    assert V.detect(file) is V.get("klein-4b-base")


def test_an_unreadable_file_is_simply_not_flux2(tmp_path: Path) -> None:
    junk = tmp_path / "notamodel.safetensors"
    junk.write_bytes(b"\x00" * 64)
    assert V.detect(junk) is None
    assert V.detect(tmp_path / "missing.safetensors") is None
