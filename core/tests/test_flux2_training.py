"""The FLUX.2 LoRA training arch: what it adapts, what it predicts, and which base it demands.

The rule these pin is the one that decides whether a run is worth anything: FLUX.2 LoRAs are trained
on an **undistilled** checkpoint and then loaded onto the distilled build for generation. Training
against a step-distilled base is the documented cause of collapse, and it fails silently - hours
later, as a bad adapter - so it is refused up front.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.training import arch as archs
from tests.test_flux2_resolve import _write_header_only
from tests.test_flux2_variants import KLEIN_4B, _shapes

models = pytest.importorskip("inline_core.training.models")


def test_flux2_is_a_registered_training_arch() -> None:
    a = archs.get("flux2")
    assert a.key == archs.FLUX2
    # Rectified flow, Krea 2's convention: x_t = (1-s)*clean + s*noise, so d/ds is noise - clean.
    assert a.target(clean=2.0, noise=5.0) == 3.0
    assert a.timestep(0.25) == 0.25


def test_targets_cover_both_block_types_and_match_the_real_model() -> None:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    with torch.device("meta"):
        model = diffusers.Flux2Transformer2DModel(**KLEIN_4B)
    linears = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    targets = archs.get("flux2").target_modules
    for target in targets:
        assert any(name.endswith(target) for name in linears), f"{target} matches nothing"
    # The single blocks hold most of the parameters; adapting only double blocks would waste them.
    assert any("to_qkv_mlp_proj" in t for t in targets)


def test_attention_scope_excludes_the_fused_single_block_projection() -> None:
    # to_qkv_mlp_proj fuses attention and MLP, so admitting it would make "attention" mean
    # "everything" on FLUX.2. The scope narrows to the double blocks instead.
    narrowed = archs.target_modules(archs.get("flux2"), "attention")
    assert "to_qkv_mlp_proj" not in narrowed
    assert {"to_q", "to_k", "to_v", "to_out.0"}.issubset(set(narrowed))
    assert set(narrowed) < set(archs.get("flux2").target_modules)


def test_the_loader_arch_is_resolved_per_variant(tmp_path: Path, monkeypatch) -> None:
    # Training says "flux2"; the loaders key their config bundles per variant, because a 4B and a
    # 9B need different encoder configs.
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    from inline_core.models.flux2 import requirements as reqs

    reqs._IDENTIFIED.clear()
    _write_header_only(
        root / "diffusion_models" / "flux-2-klein-base-4b.safetensors", _shapes(KLEIN_4B)
    )
    assert models.loader_arch("flux2") == "flux2-klein-4b"
    assert models.loader_arch("z-image") == "z-image"


def test_a_distilled_checkpoint_is_refused_with_a_pointer_to_the_base(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_FLUX2_MODEL", raising=False)
    from inline_core.models.flux2 import requirements as reqs

    reqs._IDENTIFIED.clear()
    _write_header_only(
        root / "diffusion_models" / "flux-2-klein-4b.safetensors", _shapes(KLEIN_4B)
    )
    with pytest.raises(RuntimeError, match="step-distilled"):
        models._base_file(root, archs.FLUX2, "raw")

    # Add the base build and it is chosen, even though the distilled one sorts first.
    base = _write_header_only(
        root / "diffusion_models" / "flux-2-klein-base-4b.safetensors", _shapes(KLEIN_4B)
    )
    assert models._base_file(root, archs.FLUX2, "raw") == str(base)


def test_an_empty_models_dir_says_where_to_get_a_checkpoint(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_FLUX2_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="model popup"):
        models._base_file(root, archs.FLUX2, "raw")


def test_flux2_has_no_de_distillation_adapter(tmp_path: Path) -> None:
    # Z-Image and Krea 2 offer a turbo+adapter path; FLUX.2's answer is a Base checkpoint.
    with pytest.raises(RuntimeError, match="no de-distillation adapter"):
        models._adapter_path(tmp_path, archs.FLUX2, "turbo_adapter")
    assert models._adapter_path(tmp_path, archs.FLUX2, "raw") is None


def test_flux2_can_train_in_4bit() -> None:
    from inline_core.device.policy import Quantization

    assert archs.FLUX2 in models._QUANTIZABLE
    assert models.resolve_quant("nf4", "/nonexistent", archs.FLUX2, "raw", 512) is Quantization.NF4


#: A miniature FLUX.2 with the real topology - both block types, the fused single-block projection -
#: so shapes and adapter attachment are exercised without loading a 4B checkpoint.
_TINY = {
    "attention_head_dim": 32,
    "axes_dims_rope": [8, 8, 8, 8],
    "eps": 1e-06,
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 192,
    "mlp_ratio": 3.0,
    "num_attention_heads": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "out_channels": None,
    "patch_size": 1,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}


def _tiny_model():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    return diffusers.Flux2Transformer2DModel(**_TINY).to(torch.float32).eval()


def test_one_training_step_produces_a_prediction_shaped_like_its_target() -> None:
    """The whole loop in miniature: precached latent -> noise -> forward -> loss.

    This is what catches a packing mistake. FLUX.2's VAE patchifies 2x2 on top of its 8x downscale,
    so a 512px image trains as 128 channels at 32x32, and the transformer's packing is a plain
    flatten - not the 2x2 permute other architectures in this repo use.
    """
    torch = pytest.importorskip("torch")
    from diffusers import Flux2KleinPipeline

    model = _tiny_model()
    a = archs.get("flux2")

    raw = torch.randn(1, 32, 64, 64)  # what the VAE emits for 512px
    clean = Flux2KleinPipeline._patchify_latents(raw).squeeze(0)
    assert tuple(clean.shape) == (128, 32, 32), "patchify is part of the trained representation"

    noise = torch.randn_like(clean)
    sigma = a.sigma("cpu", 3.0)
    noisy = (1 - sigma) * clean + sigma * noise
    item = {"embed": torch.randn(64, 192)}

    with torch.no_grad():
        pred = a.forward(model, noisy, a.timestep(sigma), item)
    assert pred.shape == a.target(clean, noise).shape
    assert torch.isfinite(pred).all()


def test_a_lora_adapter_attaches_to_every_target_and_receives_gradient() -> None:
    """Targets that match nothing train nothing, and PEFT does not complain loudly about it."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from diffusers import Flux2KleinPipeline

    model = _tiny_model()
    a = archs.get("flux2")
    model.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, target_modules=a.target_modules, init_lora_weights=False)
    )
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "the adapter attached to nothing"
    assert all("lora" in n for n, _ in trainable), "only the adapter should train"

    raw = torch.randn(1, 32, 32, 32)
    clean = Flux2KleinPipeline._patchify_latents(raw).squeeze(0)
    noise = torch.randn_like(clean)
    sigma = a.sigma("cpu", 3.0)
    noisy = (1 - sigma) * clean + sigma * noise
    pred = a.forward(model, noisy, a.timestep(sigma), {"embed": torch.randn(32, 192)})
    torch.nn.functional.mse_loss(pred.float(), a.target(clean, noise).float()).backward()

    got_grad = [n for n, p in trainable if p.grad is not None and p.grad.abs().sum() > 0]
    assert got_grad, "no adapter parameter received gradient"
    # Both block types must learn: the single blocks hold most of FLUX.2's parameters.
    assert any("transformer_blocks." in n and "single" not in n for n in got_grad)
    assert any("single_transformer_blocks." in n for n in got_grad)
