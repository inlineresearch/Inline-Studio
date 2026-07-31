"""The ported Fun ControlNet Union: a VACE-style side branch hooked onto a stock transformer.

The port has to match the checkpoint exactly - there is no partial credit, a mismapped tensor just
produces a subtly wrong image. These pin the structure, the zero-init no-op property that proves the
wiring is sound, and that the hooks come off again (a leaked hook would silently steer every later
render on the cached pipeline).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from inline_core.models.flux2 import controlnet as cn  # noqa: E402

#: A miniature dev: 8 double blocks (so layers 0/2/4/6 exist) and the same block topology.
TINY = {
    "attention_head_dim": 32,
    "axes_dims_rope": [8, 8, 8, 8],
    "eps": 1e-06,
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 192,
    "mlp_ratio": 3.0,
    "num_attention_heads": 4,
    "num_layers": 8,
    "num_single_layers": 2,
    "out_channels": None,
    "patch_size": 1,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}
INNER = TINY["num_attention_heads"] * TINY["attention_head_dim"]


def _model():
    return diffusers.Flux2Transformer2DModel(**TINY).eval()


def _branch():
    return cn.Flux2ControlBranch(
        inner_dim=INNER,
        num_attention_heads=TINY["num_attention_heads"],
        attention_head_dim=TINY["attention_head_dim"],
    )


def _call(model_inputs: int = 16):
    from diffusers import Flux2KleinPipeline as P

    lat = torch.randn(1, 128, model_inputs, model_inputs)
    packed = P._pack_latents(lat)
    emb = torch.randn(1, 32, 192)
    return packed, dict(
        hidden_states=packed,
        encoder_hidden_states=emb,
        timestep=torch.tensor([0.5]),
        img_ids=P._prepare_latent_ids(lat),
        txt_ids=P._prepare_text_ids(emb),
        guidance=None,
        return_dict=False,
    )


def test_the_branch_matches_the_published_checkpoint_layout() -> None:
    """The union ships 4 control blocks plus one input projection; before_proj is block 0 only."""
    branch = _branch()
    keys = set(branch.state_dict())
    assert len(branch.control_transformer_blocks) == 4 == len(cn.CONTROL_LAYERS)
    assert cn.CONTROL_IN_DIM == 260, "control latent (128) + mask (4) + inpaint latent (128)"
    assert any(k.startswith("control_img_in.") for k in keys)
    assert any(k.startswith("before_proj.") for k in keys)
    assert sum(1 for k in keys if k.startswith("after_proj.") and k.endswith(".weight")) == 4


def test_checkpoint_keys_remap_onto_the_branch() -> None:
    """Upstream nests the projections inside each control block; we keep the blocks stock."""
    remapped = cn._remap(
        {
            "control_img_in.weight": 1,
            "control_transformer_blocks.0.before_proj.weight": 2,
            "control_transformer_blocks.2.after_proj.bias": 3,
            "control_transformer_blocks.1.attn.to_q.weight": 4,
        }
    )
    assert remapped == {
        "control_img_in.weight": 1,
        "before_proj.weight": 2,
        "after_proj.2.bias": 3,
        "control_transformer_blocks.1.attn.to_q.weight": 4,
    }


def test_a_zero_init_branch_changes_nothing() -> None:
    """The correctness invariant: before_proj and after_proj are zero-init, so an unloaded branch
    is exactly the base model. If this drifts, the injection is wired wrong."""
    model, branch = _model(), _branch()
    packed, call = _call()
    context = torch.randn(1, packed.shape[1], cn.CONTROL_IN_DIM)
    with torch.no_grad():
        base = model(**call)[0]
        handles = cn.attach(model, branch, context, scale=0.75)
        try:
            hooked = model(**call)[0]
        finally:
            for handle in handles:
                handle.remove()
    assert torch.allclose(base, hooked, atol=1e-5)


def test_hooks_detach_completely() -> None:
    """A leaked hook would steer every later render on the cached pipeline."""
    model, branch = _model(), _branch()
    for tap in branch.after_proj:
        torch.nn.init.normal_(tap.weight, std=0.02)
    packed, call = _call()
    context = torch.randn(1, packed.shape[1], cn.CONTROL_IN_DIM)
    with torch.no_grad():
        base = model(**call)[0]
        handles = cn.attach(model, branch, context, scale=1.0)
        steered = model(**call)[0]
        for handle in handles:
            handle.remove()
        restored = model(**call)[0]
    assert not torch.allclose(base, steered, atol=1e-4), "a trained branch must steer"
    assert torch.allclose(base, restored, atol=1e-6), "removal must restore the base exactly"


def test_strength_scales_the_residual() -> None:
    model, branch = _model(), _branch()
    for tap in branch.after_proj:
        torch.nn.init.normal_(tap.weight, std=0.02)
    packed, call = _call()
    context = torch.randn(1, packed.shape[1], cn.CONTROL_IN_DIM)

    def run(scale: float):
        handles = cn.attach(model, branch, context, scale=scale)
        try:
            with torch.no_grad():
                return model(**call)[0]
        finally:
            for handle in handles:
                handle.remove()

    base, half, full = run(0.0), run(0.5), run(1.0)
    assert torch.allclose(base, run(0.0), atol=1e-6)
    # Monotonic: more strength moves further from the unsteered result.
    assert (half - base).abs().mean() < (full - base).abs().mean()


def test_a_model_without_enough_blocks_is_refused() -> None:
    """The union is built for dev's 8 double blocks; klein has 5 and must not load it."""
    small = diffusers.Flux2Transformer2DModel(**{**TINY, "num_layers": 5}).eval()
    from inline_core.errors import ComponentError

    with pytest.raises(ComponentError, match="built for FLUX.2 dev"):
        cn.attach(small, _branch(), torch.zeros(1, 4, cn.CONTROL_IN_DIM))
