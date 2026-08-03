"""The key plan and, more importantly, the layout detector standing in front of it.

Splitting a fused QKV the wrong way round fills every parameter, passes every shape check and
renders a video that plays. So the detector gets tested on synthetic tensors whose layout is known
by construction, and - opt-in, because it needs the network - against the two real MiniMax H3
checkpoints, one of which is per-head interleaved and the other the same data de-interleaved.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess

import numpy as np
import pytest

from inline_core.errors import ComponentError
from inline_core.models.keymap import (
    AssertEqual,
    Drop,
    KeyPlan,
    RowLayout,
    Split,
    SwapHalves,
    check_coverage,
    detect_row_layout,
    row_stats,
    transform,
)

PARTS, HEADS, HEAD_DIM, COLS = 3, 8, 4, 6
ROWS = PARTS * HEADS * HEAD_DIM


def _parts(scale: tuple[float, float, float]) -> list[np.ndarray]:
    """Three blocks with genuinely different magnitudes, as q, k and v have."""
    rng = np.random.default_rng(0)
    return [
        (rng.standard_normal((HEADS * HEAD_DIM, COLS)) * s).astype(np.float32) for s in scale
    ]


def _contiguous(scale: tuple[float, float, float] = (1.0, 0.6, 0.25)) -> np.ndarray:
    return np.concatenate(_parts(scale), axis=0)


def _interleaved(scale: tuple[float, float, float] = (1.0, 0.6, 0.25)) -> np.ndarray:
    blocks = _parts(scale)
    rows = []
    for head in range(HEADS):
        for part in blocks:
            rows.append(part[head * HEAD_DIM : (head + 1) * HEAD_DIM])
    return np.concatenate(rows, axis=0)


# --- the detector ---------------------------------------------------------------------------------


def test_detector_names_each_synthetic_layout() -> None:
    assert detect_row_layout(row_stats(_contiguous()), PARTS, HEAD_DIM) is RowLayout.CONTIGUOUS
    assert detect_row_layout(row_stats(_interleaved()), PARTS, HEAD_DIM) is RowLayout.INTERLEAVED


def test_detector_holds_up_when_the_parts_barely_differ() -> None:
    """The FFN halves sit ~4 percent apart, against ~30 for q/k/v."""
    close = np.concatenate(
        [
            (np.random.default_rng(1).standard_normal((48, COLS)) * 1.00).astype(np.float32),
            (np.random.default_rng(2).standard_normal((48, COLS)) * 1.04).astype(np.float32),
        ]
    )
    assert detect_row_layout(row_stats(close), 2, 4) is RowLayout.CONTIGUOUS


def test_detector_rejects_dimensions_that_do_not_divide() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        detect_row_layout([1.0] * 10, 3, 4)


# --- the transforms -------------------------------------------------------------------------------


def test_split_slices_a_contiguous_source_in_order() -> None:
    tensor = _contiguous()
    plan = Split(("q", "k", "v"), layout=RowLayout.CONTIGUOUS, head_dim=HEAD_DIM)

    out = dict(transform("qkv", tensor, plan))

    assert list(out) == ["q", "k", "v"]
    assert np.array_equal(out["q"], tensor[: HEADS * HEAD_DIM])
    assert np.array_equal(out["v"], tensor[2 * HEADS * HEAD_DIM :])


def test_split_deinterleaves_before_slicing() -> None:
    """The whole point: an interleaved source must come out as the same three parts."""
    expected = dict(
        zip(
            ("q", "k", "v"),
            np.split(_contiguous(), PARTS),
            strict=True,
        )
    )
    plan = Split(("q", "k", "v"), layout=RowLayout.INTERLEAVED, head_dim=HEAD_DIM)

    out = dict(transform("qkv", _interleaved(), plan))

    for name in ("q", "k", "v"):
        assert np.array_equal(out[name], expected[name]), name


def test_split_refuses_a_source_in_the_other_layout() -> None:
    """A plan written for one repack, pointed at another, must fail loudly rather than render."""
    plan = Split(("q", "k", "v"), layout=RowLayout.CONTIGUOUS, head_dim=HEAD_DIM)
    with pytest.raises(ComponentError, match="looks interleaved"):
        list(transform("qkv", _interleaved(), plan))


def test_swap_halves_exchanges_gate_and_value() -> None:
    tensor = np.concatenate([np.zeros((4, 3), np.float32), np.ones((4, 3), np.float32)])
    (_, swapped), = transform("fc1", tensor, SwapHalves("ff.net.0.proj"))
    assert np.array_equal(swapped[:4], np.ones((4, 3), np.float32))
    assert np.array_equal(swapped[4:], np.zeros((4, 3), np.float32))


def test_swap_halves_refuses_an_odd_tensor() -> None:
    with pytest.raises(ComponentError, match="odd first dimension"):
        list(transform("fc1", np.zeros((5, 3), np.float32), SwapHalves("x")))


def test_dropped_and_asserted_tensors_yield_nothing() -> None:
    assert list(transform("k", np.zeros((2, 2)), Drop("recomputed"))) == []
    assert list(transform("k", np.zeros((2, 2)), AssertEqual("rope.inv_freq"))) == []


# --- coverage -------------------------------------------------------------------------------------


def _plan() -> KeyPlan:
    return KeyPlan(
        version="v1",
        actions={
            "blocks.0.attn.qkv_proj.weight": Split(
                ("blocks.0.attn.to_q.weight", "blocks.0.attn.to_k.weight",
                 "blocks.0.attn.to_v.weight"),
            ),
            "blocks.0.mlp.fc1.weight": SwapHalves("blocks.0.ff.net.0.proj.weight"),
            "rope.inv_freq": AssertEqual("rope.inv_freq"),
        },
    )


def test_coverage_accepts_a_complete_plan() -> None:
    check_coverage(
        _plan(),
        source_keys=["blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight", "rope.inv_freq"],
        target_keys=sorted(_plan().targets()),
    )


def test_coverage_names_an_unmapped_checkpoint_tensor() -> None:
    with pytest.raises(ComponentError, match="no action in key plan"):
        check_coverage(_plan(), ["blocks.0.attn.qkv_proj.weight", "surprise.weight"], [])


def test_coverage_names_an_unfilled_model_parameter() -> None:
    with pytest.raises(ComponentError, match="left unfilled"):
        check_coverage(
            _plan(),
            ["blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight", "rope.inv_freq"],
            [*_plan().targets(), "blocks.0.attn.to_out.weight"],
        )


def test_coverage_names_a_plan_entry_the_checkpoint_lacks() -> None:
    with pytest.raises(ComponentError, match="does not have"):
        check_coverage(_plan(), ["blocks.0.attn.qkv_proj.weight"], [])


# --- against the real checkpoints (opt-in: needs the network) -------------------------------------

_HF = "https://huggingface.co/{}/resolve/main/{}"
_H3 = {
    RowLayout.INTERLEAVED: _HF.format(
        "MiniMaxAI/MiniMax-H3", "FL2VA/transformer/model-00001-of-00013.safetensors"
    ),
    RowLayout.CONTIGUOUS: _HF.format(
        "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_bf16.safetensors"
    ),
}
_INT8 = _HF.format(
    "Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"
)
_KEY = "blocks.0.attn.qkv_proj.weight"
#: H3's geometry, from FL2VA/transformer/config.json: 56 heads of 128 into a 5376-wide model.
_H3_HEADS, _H3_HEAD_DIM, _H3_COLS = 56, 128, 5376

needs_network = pytest.mark.skipif(
    not os.environ.get("INLINE_NETWORK_TESTS"),
    reason="set INLINE_NETWORK_TESTS=1 to check against the published checkpoints",
)


def _fetch_rows(
    url: str, key: str, first_row: int, rows: int, cols: int = _H3_COLS, dtype: str = "bf16"
) -> np.ndarray:
    """Rows ``first_row`` onward of a 2D tensor, by header offset.

    Rows are contiguous in safetensors, so this is one range request rather than a download.
    """
    head = subprocess.run(
        ["curl", "-sfL", "-r", "0-200000", url], capture_output=True, check=True
    ).stdout
    size = struct.unpack("<Q", head[:8])[0]
    header = json.loads(head[8 : 8 + size])
    width = 4 if dtype == "f32" else 2
    start = 8 + size + header[key]["data_offsets"][0] + first_row * cols * width
    raw = subprocess.run(
        ["curl", "-sfL", "-r", f"{start}-{start + rows * cols * width - 1}", url],
        capture_output=True, check=True,
    ).stdout
    if dtype == "f32":
        return np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)
    # bf16 is the top two bytes of an fp32, so widening is a shift, not a conversion table.
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32).reshape(rows, cols)


@needs_network
def test_deinterleave_reproduces_the_published_deinterleaved_checkpoint() -> None:
    """The transform against ground truth.

    MiniMaxAI ships this tensor per-head interleaved and Comfy-Org ships the same data already
    de-interleaved, so our de-interleave has to turn one into the other exactly. Two heads, because
    with one head the two layouts are the same arrangement and the test would pass trivially.
    """
    heads = 2
    interleaved = _fetch_rows(_H3[RowLayout.INTERLEAVED], _KEY, 0, PARTS * heads * _H3_HEAD_DIM)
    plan = Split(("q", "k", "v"), layout=RowLayout.INTERLEAVED, head_dim=_H3_HEAD_DIM)

    out = dict(transform(_KEY, interleaved, plan, verify_layout=False))

    span = heads * _H3_HEAD_DIM
    for index, name in enumerate(("q", "k", "v")):
        # Each part begins at its own third of the de-interleaved file: rows 0, 7168 and 14336.
        expected = _fetch_rows(
            _H3[RowLayout.CONTIGUOUS], _KEY, index * _H3_HEADS * _H3_HEAD_DIM, span
        )
        assert np.array_equal(out[name], expected), f"{name} does not match the published file"


@needs_network
def test_detector_reads_the_real_layout_off_a_full_length_statistic() -> None:
    """The published int8 build's per-row scales cover all 21504 rows for 86 KB, which is the
    whole-tensor measurement the detector is meant to be handed."""
    rows = _H3_HEADS * _H3_HEAD_DIM * PARTS
    scales = _fetch_rows(_INT8, f"{_KEY}_scale", 0, rows, cols=1, dtype="f32")

    assert detect_row_layout([float(v) for v in scales[:, 0]], PARTS, _H3_HEAD_DIM) is (
        RowLayout.CONTIGUOUS
    )


def test_detector_refuses_a_single_head_rather_than_guessing() -> None:
    """A window off the front of a tensor cannot answer this, and must not appear to."""
    with pytest.raises(ValueError, match="one head"):
        detect_row_layout([1.0] * (PARTS * 128), PARTS, 128)
