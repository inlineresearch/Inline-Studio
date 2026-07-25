"""The byte-range checkpoint reader.

Why it exists: safetensors maps a whole file at once, and Linux refuses a mapping larger than
physical RAM when there is no swap - so a 26GB Krea 2 checkpoint cannot be opened at all on a 16GB
machine, whatever the model would cost once quantized. Reading byte ranges sidesteps that, so this
must agree with safetensors exactly.
"""

from __future__ import annotations

import pytest

from inline_core.errors import ComponentError
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
