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


def test_the_build_satisfies_the_vendored_depth_guard() -> None:
    """`encoders.py` refuses a conditioner whose config reports 50 layers, because a stack naively
    truncated there makes `hidden_states[50]` post-norm. This build carries exactly 50 and defeats
    that by swapping the trailing norm for Identity, so the state stays pre-norm - and the config
    has to report the depth it was cut from or the guard rejects a correct encoder."""
    import pathlib

    from inline_core.models.minimaxh3.vendor.packing import MINIMAX_H3_TEXT_ENCODER_LAYER

    weights = pathlib.Path("models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    if not weights.is_file():
        pytest.skip("nvfp4 conditioner not present")

    from inline_core.models.minimaxh3.pipeline import _load_nvfp4_encoder

    model = _load_nvfp4_encoder(weights, torch.bfloat16)
    depth = model.config.text_config.num_hidden_layers
    assert depth > MINIMAX_H3_TEXT_ENCODER_LAYER, "the vendored guard would reject this"
    # The stack really is the file's, whatever the config says: transformers reads the config only
    # when constructing and iterates the module list at inference.
    assert len(model.model.language_model.layers) == MINIMAX_H3_TEXT_ENCODER_LAYER
    assert isinstance(model.model.language_model.norm, torch.nn.Identity)
    assert isinstance(model.lm_head, torch.nn.Identity), "this build ships no head"


def test_the_vae_budget_counts_the_denoiser_that_lands_after_it(monkeypatch) -> None:
    """`_vae_fits` runs three lines before `apply_offload`, so the card it measures is empty and
    every later placement is invisible. Reserving against that put a 10.4 GB fp32 VAE beside a
    33 GB denoiser on a 44 GB card, and the render peaked at 43.4 GB and died."""
    import pathlib

    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3 import pipeline as pl

    vae = pathlib.Path("models/vae/minimax_h3_video_vae_fp16.safetensors")
    if not vae.is_file():
        pytest.skip("video VAE not present")

    monkeypatch.setattr(rt, "free_vram_bytes", lambda *a, **k: int(46.6e9))
    assert pl._vae_fits(vae, None, 0), "an empty card looks like room, which is the old bug"
    assert not pl._vae_fits(vae, None, int(33e9)), "counting the denoiser has to flip it"

    # The fast path survives where the card genuinely has room: leaf offload turned a 6 minute
    # render into 32, so this must not become "always stream".
    monkeypatch.setattr(rt, "free_vram_bytes", lambda *a, **k: int(80e9))
    assert pl._vae_fits(vae, None, int(33e9))

    # A denoiser larger than the whole card is a decision, not a crash.
    monkeypatch.setattr(rt, "free_vram_bytes", lambda *a, **k: int(40e9))
    assert not pl._vae_fits(vae, None, int(80e9))


def test_a_resident_denoiser_is_sized_whole() -> None:
    """With no offload every byte of it lands on the card, so that is what the VAE must budget."""
    from dataclasses import dataclass

    from inline_core.models.minimaxh3.pipeline import _denoiser_card_bytes

    @dataclass
    class _Recipe:
        denoiser_offload: object = None

    model = torch.nn.Linear(64, 32, bias=False, dtype=torch.float32)
    assert _denoiser_card_bytes(model, _Recipe(), 0) == 64 * 32 * 4


def test_the_forward_is_correct_at_a_reference_length_sequence() -> None:
    """The chunked matmul writes into a preallocated output, so a wrong slice bound would corrupt
    part of it rather than raising. Five references put ~20k tokens through this, and the short
    prompt the earlier tests use would not catch an off-by-one in the row bounds."""
    weight = _weight(96, 64)
    packed, block_scale, global_scale = nvfp4.quantize_reference(weight)

    layer = nvfp4.NVFP4Linear(64, 96, bias=True, dtype=torch.float32)
    layer.weight.copy_(packed)
    layer.weight_scale.copy_(block_scale)
    layer.weight_scale_2.copy_(global_scale)
    bias = torch.randn(96, dtype=torch.float32) * 0.01
    with torch.no_grad():
        layer.bias.copy_(bias)

    x = torch.randn(1, 512, 64, dtype=torch.float32)
    got = layer(x)
    want = torch.nn.functional.linear(x, weight, bias)
    assert got.shape == want.shape == (1, 512, 96)
    # Every column must be right, not just the aggregate: a bad chunk bound leaves a correct-looking
    # tensor with one band of zeros, which a whole-tensor cosine hides.
    per_column = torch.nn.functional.cosine_similarity(got, want, dim=1)
    assert float(per_column.min()) > 0.99, float(per_column.min())


def test_the_output_is_allocated_once_not_concatenated(monkeypatch) -> None:
    """`torch.cat` holds every chunk and the result at the same time. At 20k tokens that is an
    extra gigabyte, which is what it ran out of. Asserted by watching for the call rather than
    grepping the source, which matched the comment explaining why it is gone."""
    weight = _weight(96, 64)
    packed, block_scale, global_scale = nvfp4.quantize_reference(weight)
    layer = nvfp4.NVFP4Linear(64, 96, bias=False, dtype=torch.float32)
    layer.weight.copy_(packed)
    layer.weight_scale.copy_(block_scale)
    layer.weight_scale_2.copy_(global_scale)

    calls: list[int] = []
    real_cat = torch.cat
    monkeypatch.setattr(torch, "cat", lambda *a, **k: (calls.append(1), real_cat(*a, **k))[1])

    out = layer(torch.randn(1, 256, 64, dtype=torch.float32))
    assert out.shape == (1, 256, 96)
    assert not calls, "the chunks are written into one output, never concatenated"
