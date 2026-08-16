"""The byte-range checkpoint reader.

Why it exists: safetensors maps a whole file at once, and Linux refuses a mapping larger than
physical RAM when there is no swap - so a 26GB Krea 2 checkpoint cannot be opened at all on a 16GB
machine, whatever the model would cost once quantized. Reading byte ranges sidesteps that, so this
must agree with safetensors exactly.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from inline_core.errors import ComponentError
from inline_core.models import checkpoint
from inline_core.models.checkpoint import CheckpointReader

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")


@pytest.fixture
def written(tmp_path):
    """A file covering the dtypes and shapes a real checkpoint mixes."""
    state = {
        "big.weight": torch.randn(64, 32, dtype=torch.float32),
        "half.weight": torch.randn(16, 8).to(torch.bfloat16),
        "flat": torch.arange(24, dtype=torch.float32),
        "scalar": torch.tensor(3.5),
        "bias": torch.zeros(7, dtype=torch.float16),
    }
    path = tmp_path / "model.safetensors"
    safetensors.save_file(state, str(path), metadata={"format": "pt"})
    return path, state


def test_every_tensor_matches_safetensors(written) -> None:
    path, state = written

    with CheckpointReader(path) as reader:
        assert sorted(reader.keys()) == sorted(state)
        for key, expected in state.items():
            got = reader.get_tensor(key)
            assert got.dtype == expected.dtype, key
            assert got.shape == expected.shape, key
            assert torch.equal(got.float(), expected.float()), key


def test_metadata_is_exposed_and_not_mistaken_for_a_tensor(written) -> None:
    path, _ = written

    reader = CheckpointReader(path)

    assert "__metadata__" not in reader.keys()
    assert reader.metadata.get("format") == "pt"


def test_reads_the_same_bytes_as_safetensors_for_a_large_offset(written) -> None:
    path, state = written

    # The last tensor by offset is the one a naive reader gets wrong.
    with CheckpointReader(path) as reader:
        from_range = {k: reader.get_tensor(k) for k in reader.keys()}
    from_mmap = safetensors.load_file(str(path))

    assert set(from_range) == set(from_mmap)
    for key, tensor in from_range.items():
        assert torch.equal(tensor.float(), from_mmap[key].float()), key


def test_a_non_checkpoint_file_is_a_clear_error(tmp_path) -> None:
    path = tmp_path / "not-a-checkpoint.safetensors"
    path.write_bytes(b"this is not a safetensors file at all")

    with pytest.raises(ComponentError, match="Could not read checkpoint"):
        CheckpointReader(path)


def test_a_missing_file_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(ComponentError, match="Could not read checkpoint"):
        CheckpointReader(tmp_path / "absent.safetensors")


# --- already-quantized checkpoints ---------------------------------------------------------------
#
# A ComfyUI-quantized text encoder loads into a stock Qwen3Model with its `comfy_quant` and
# `weight_scale` keys dropped as unexpected, leaving packed tensors in layers sized for unpacked
# ones. That surfaced as a matmul shape error deep inside torchao, after the pipeline had
# already reported itself ready.


def _write(path: Path, index: dict[str, object]) -> Path:
    blob = json.dumps(index).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 64)
    return path


def _t(dtype: str, shape: list[int]) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [0, 1]}


def test_a_comfy_quantized_encoder_is_recognised(tmp_path: Path) -> None:
    file = _write(
        tmp_path / "qwen_3_8b_comfyquant.safetensors",
        {
            "model.layers.0.self_attn.k_proj.weight": _t("I8", [1024, 2048]),
            "model.layers.0.self_attn.k_proj.weight_scale": _t("F32", [1024]),
            "model.layers.0.self_attn.k_proj.comfy_quant": _t("I8", []),
        },
    )
    assert checkpoint.prequantized_kind(file) == "prequantized"


def test_an_fp8_checkpoint_is_recognised(tmp_path: Path) -> None:
    file = _write(
        tmp_path / "enc_fp8.safetensors",
        {"model.layers.0.mlp.up_proj.weight": _t("F8_E4M3", [4096, 4096])},
    )
    assert checkpoint.prequantized_kind(file) == "fp8"


def test_a_plain_checkpoint_is_not_flagged(tmp_path: Path) -> None:
    """A false positive here refuses to quantize a model that needs it, so it must not fire on
    ordinary weights or on the RMSNorm tensors every transformer carries."""
    file = _write(
        tmp_path / "qwen_3_8b.safetensors",
        {
            "model.layers.0.self_attn.k_proj.weight": _t("BF16", [1024, 4096]),
            "model.layers.0.input_layernorm.weight": _t("BF16", [4096]),
            "model.layers.0.self_attn.q_norm.scale": _t("BF16", [128]),
        },
    )
    assert checkpoint.prequantized_kind(file) is None


def test_an_unreadable_file_is_not_classified(tmp_path: Path) -> None:
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"\x00" * 32)
    assert checkpoint.prequantized_kind(junk) is None
    assert checkpoint.prequantized_kind(tmp_path / "missing.safetensors") is None
