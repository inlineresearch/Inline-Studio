"""NVFP4 unpacking, against ComfyUI's own quantiser rather than against our own assumptions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from inline_core.models.minimaxh3 import nvfp4  # noqa: E402


def _weight(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, dtype=torch.float32) * 0.02


def test_the_swizzle_round_trips() -> None:
    """`from_blocked` has to invert `to_blocked` exactly; a near-miss still scores ~0.9 against a
    real checkpoint, which reads as "close enough" and is not."""
    scales = torch.arange(128 * 320, dtype=torch.float32).reshape(128, 320)
    assert torch.equal(nvfp4.from_blocked(nvfp4.to_blocked(scales), 128, 320), scales)


def test_the_swizzle_round_trips_across_several_tiles() -> None:
    for rows, cols in ((256, 320), (384, 64), (128, 4)):
        scales = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        assert torch.equal(nvfp4.from_blocked(nvfp4.to_blocked(scales), rows, cols), scales)


def test_the_swizzle_is_not_the_identity() -> None:
    """Guards the test above: if `to_blocked` were a no-op both would pass and prove nothing."""
    scales = torch.arange(128 * 320, dtype=torch.float32).reshape(128, 320)
    assert not torch.equal(nvfp4.to_blocked(scales), scales)


def test_unpacking_is_high_nibble_first() -> None:
    table = nvfp4.e2m1_table()
    # 0x17 -> high 1 (0.5), low 7 (6.0). The reversed reading scores ~0 on a real checkpoint.
    got = nvfp4.unpack_fp4(torch.tensor([[0x17]], dtype=torch.uint8), table)
    assert got.tolist() == [[0.5, 6.0]]
    # High bit is the sign, so 0x8F is -0.0 then -6.0.
    assert nvfp4.unpack_fp4(torch.tensor([[0x8F]], dtype=torch.uint8), table).tolist() == [
        [-0.0, -6.0]
    ]


def test_a_quantised_weight_dequantises_back() -> None:
    """The whole contract: ComfyUI packs, we unpack, and the result is the same weight to within
    4-bit block quantisation error."""
    weight = _weight(256, 512)
    packed, block_scale, global_scale = nvfp4.quantize_reference(weight)
    assert packed.shape == (256, 256)
    assert packed.dtype == torch.uint8
    assert block_scale.dtype == torch.float8_e4m3fn

    got = nvfp4.dequantize(
        packed, block_scale, global_scale,
        out_features=256, in_features=512, dtype=torch.float32,
    )
    assert got.shape == weight.shape
    cosine = torch.nn.functional.cosine_similarity(got.flatten(), weight.flatten(), dim=0)
    assert cosine > 0.99, cosine
    # Block-16 scaling keeps the magnitude, which is what a wrong scale convention destroys first.
    assert 0.9 < got.std() / weight.std() < 1.1


def test_skipping_the_swizzle_is_visibly_wrong() -> None:
    """Records the bug that cost the most time: reading the scales in stored order still produces a
    plausible-looking weight, so nothing short of comparing against a reference catches it."""
    weight = _weight(256, 512)
    packed, block_scale, global_scale = nvfp4.quantize_reference(weight)
    table = nvfp4.e2m1_table()
    values = nvfp4.unpack_fp4(packed, table)
    naive = (
        values.reshape(256, 32, 16) * block_scale.to(torch.float32).unsqueeze(-1)
    ).reshape(256, 512) * float(global_scale)
    cosine = torch.nn.functional.cosine_similarity(naive.flatten(), weight.flatten(), dim=0)
    assert cosine < 0.95, "the un-swizzled read should be clearly wrong, not subtly wrong"


def test_the_linear_matches_an_ordinary_one() -> None:
    weight = _weight(128, 256)
    bias = torch.randn(128, dtype=torch.float32) * 0.01
    packed, block_scale, global_scale = nvfp4.quantize_reference(weight)

    layer = nvfp4.NVFP4Linear(256, 128, bias=True, dtype=torch.float32)
    layer.weight.copy_(packed)
    layer.weight_scale.copy_(block_scale)
    layer.weight_scale_2.copy_(global_scale)
    with torch.no_grad():
        layer.bias.copy_(bias)

    x = torch.randn(4, 256, dtype=torch.float32)
    got = layer(x)
    want = torch.nn.functional.linear(x, weight, bias)
    assert got.shape == want.shape
    assert torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0) > 0.99


def test_the_weight_is_never_materialised_on_the_module() -> None:
    """Holding the unpacked weight would defeat the point: the packed buffers are the reason a
    15.7 GB file stands in for a 63 GB folder."""
    layer = nvfp4.NVFP4Linear(256, 128, bias=False, dtype=torch.float32)
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.numel() == 128 * 128, "one byte per two weights"


def test_awq_smoothing_is_applied_to_the_input() -> None:
    """Measured on the real file: `w * pre_quant_scale` recovers the original (1.03) where dividing
    does not (0.59), so the activation is what carries the factor."""
    weight = _weight(64, 128)
    smooth = torch.rand(128, dtype=torch.float32) + 0.5
    packed, block_scale, global_scale = nvfp4.quantize_reference(weight / smooth)

    layer = nvfp4.NVFP4Linear(128, 64, bias=False, dtype=torch.float32)
    layer.weight.copy_(packed)
    layer.weight_scale.copy_(block_scale)
    layer.weight_scale_2.copy_(global_scale)
    layer.pre_quant_scale = smooth

    x = torch.randn(4, 128, dtype=torch.float32)
    got = layer(x)
    want = torch.nn.functional.linear(x, weight)
    assert torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0) > 0.99
