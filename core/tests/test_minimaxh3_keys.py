"""MiniMax H3's key plan, checked against the real checkpoint header and the real port.

The plan is what turns a checkpoint written for MiniMax's implementation into the vendored diffusers
model, and its two interesting transforms both fail silently. So the arithmetic is pinned here,
the coverage is proved against the published header (a few hundred KB of range requests, not a
66 GB download), and the FFN swap is checked against diffusers' own SwiGLU rather than a comment.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess

import pytest

from inline_core.models.keymap import AssertEqual, Rename, RowLayout, Split, SwapHalves

torch = pytest.importorskip("torch")
keys_module = pytest.importorskip("inline_core.models.minimaxh3.keys")

build_plan = keys_module.build_plan
self_computed_targets = keys_module.self_computed_targets

SOURCE_URL = (
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
    "/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
)
#: 50 transformer blocks plus 2 refiner blocks, each fusing one QKV.
ATTENTION_BLOCKS = 52
SOURCE_TENSORS = 535
TARGET_TENSORS = 639


def _plan_targets(plan: object) -> set[str]:
    return plan.targets()  # type: ignore[attr-defined]


def test_the_arithmetic_that_makes_the_plan_complete() -> None:
    """639 targets from 535 sources: each attention block turns one fused QKV into three."""
    assert SOURCE_TENSORS + ATTENTION_BLOCKS * 2 == TARGET_TENSORS


def test_every_source_tensor_has_exactly_one_action() -> None:
    plan = build_plan("comfy-org")
    assert len(plan.actions) == SOURCE_TENSORS


def test_the_two_publishers_differ_only_in_qkv_row_layout() -> None:
    """Same data, different row order, so only the Split layout may differ between them."""
    comfy = build_plan("comfy-org").actions
    minimax = build_plan("minimaxai").actions
    assert set(comfy) == set(minimax)
    differing = [k for k in comfy if comfy[k] != minimax[k]]
    assert differing and all("qkv_proj" in k for k in differing)
    assert comfy["blocks.0.attn.qkv_proj.weight"].layout is RowLayout.CONTIGUOUS
    assert minimax["blocks.0.attn.qkv_proj.weight"].layout is RowLayout.INTERLEAVED


def test_an_unknown_publisher_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="Unknown checkpoint source"):
        build_plan("some-other-repack")


def test_the_fused_tensors_get_the_transform_they_need() -> None:
    actions = build_plan("comfy-org").actions
    qkv = actions["blocks.0.attn.qkv_proj.weight"]
    assert isinstance(qkv, Split) and len(qkv.targets) == 3 and qkv.head_dim == 128
    assert isinstance(actions["blocks.0.mlp.fc1.weight"], SwapHalves)
    assert isinstance(actions["blocks.0.attn.out_proj.weight"], Rename)


def test_rope_is_asserted_rather_than_loaded_or_dropped() -> None:
    """The config omits rope_theta, so the shipped table is the ground truth to check against."""
    assert isinstance(build_plan("comfy-org").actions["rope.inv_freq"], AssertEqual)
    assert "rope.inv_freq" in self_computed_targets()


def test_the_refiner_blocks_have_no_adaln_branch() -> None:
    actions = build_plan("comfy-org").actions
    assert "blocks.0.adaln_proj.linear.weight" in actions
    assert "token_refiner.blocks.0.adaln_proj.linear.weight" not in actions


def test_diffusers_swiglu_still_reads_value_then_gate() -> None:
    """The reason `mlp.fc1` swaps halves. If upstream ever reorders this, the swap must go."""
    import inspect

    from diffusers.models.activations import SwiGLU

    source = inspect.getsource(SwiGLU.forward)
    assert "hidden_states, gate = hidden_states.chunk(2" in source
    assert "hidden_states * self.activation(gate)" in source


def test_the_plan_fills_the_real_port_exactly() -> None:
    """Build the vendored transformer on a meta device - no allocation - and check every one of its
    parameters is filled by the plan and nothing is left over."""
    from inline_core.models.minimaxh3.vendor import MiniMaxH3Transformer3DModel

    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel()
    target = set(dict(model.named_parameters()) | dict(model.named_buffers()))

    assert len(target) == TARGET_TENSORS
    plan_targets = _plan_targets(build_plan("comfy-org"))
    assert target - plan_targets == self_computed_targets()
    assert plan_targets - target == set()


@pytest.mark.skipif(
    not os.environ.get("INLINE_NETWORK_TESTS"),
    reason="set INLINE_NETWORK_TESTS=1 to check the plan against the published checkpoint",
)
def test_the_plan_covers_the_published_checkpoint() -> None:
    """Coverage against the real 535-tensor header, which costs a range request, not a download."""
    from inline_core.models.keymap import check_coverage
    from inline_core.models.minimaxh3.vendor import MiniMaxH3Transformer3DModel

    head = subprocess.run(
        ["curl", "-sfL", "-r", "0-200000", SOURCE_URL], capture_output=True, check=True
    ).stdout
    size = struct.unpack("<Q", head[:8])[0]
    header = json.loads(head[8 : 8 + size])
    header.pop("__metadata__", None)

    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel()
    targets = set(dict(model.named_parameters()) | dict(model.named_buffers()))

    check_coverage(
        build_plan("comfy-org"),
        sorted(header),
        sorted(targets - self_computed_targets()),
    )
    assert header["blocks.0.attn.qkv_proj.weight"]["shape"] == [21504, 5376]  # 3 x 56 x 128
    assert header["blocks.0.mlp.fc1.weight"]["shape"] == [28672, 5376]  # 2 x 14336
