"""NVFP4 (ComfyUI's ``comfy_quant``) weights, dequantised per layer at inference.

The file's own marker says ``{"format": "nvfp4", "full_precision_matrix_mult": true}``: it is meant
to be unpacked into an ordinary matmul, not fed to FP4 tensor cores, so it runs on any card rather
than needing Blackwell. Weights stay 4-bit in VRAM and each layer is unpacked inside its own
forward, which is what makes a 15.7 GB file stand in for a 63 GB folder.

The layout is decoded in ``docs/nvfp4-format.md`` and verified tensor-by-tensor against the bf16
release of the same encoder.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: FP4 E2M1: sign in bit 3, exponent in bits 2-1, mantissa in bit 0.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0
#: One FP8 scale per this many weights, along the input dimension.
BLOCK = 16


def e2m1_table(device: Any = None, dtype: Any = torch.float32) -> torch.Tensor:
    """The 16 representable FP4 values, indexed by nibble."""
    return torch.tensor(E2M1 + tuple(-v for v in E2M1), device=device, dtype=dtype)


def unpack_fp4(packed: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``[..., n/2]`` uint8 to ``[..., n]`` values. High nibble first, then low, interleaved.

    The other three orderings score ~0 against the reference, so this is not a coin flip.
    """
    codes = packed.to(torch.int64)
    pairs = torch.stack((table[codes >> 4], table[codes & 0xF]), dim=-1)
    return pairs.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def from_blocked(stored: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo the cuBLAS block-scaling swizzle that ``comfy/float.py``'s ``to_blocked`` applies.

    Two stages, not one, which is why searching single reshape+permute patterns never finds it:
    skipping this scores 0.899 against the reference where undoing it scores 0.991.
    """
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    tiles = stored.reshape(row_blocks * col_blocks, 32, 4, 4).transpose(1, 2)
    flat = tiles.reshape(row_blocks, col_blocks, 128, 4).permute(0, 2, 1, 3)
    return flat.reshape(row_blocks * 128, col_blocks * 4)[:rows, :cols]


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """The forward swizzle, kept so a test can prove ``from_blocked`` inverts the real thing."""
    rows, cols = scales.shape
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    padded = scales
    if (rows, cols) != (row_blocks * 128, col_blocks * 4):
        padded = torch.zeros(
            (row_blocks * 128, col_blocks * 4), device=scales.device, dtype=scales.dtype
        )
        padded[:rows, :cols] = scales
    tiles = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(row_blocks * 128, col_blocks * 4)


def dequantize(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor | float,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One NVFP4 weight back to ``dtype``: ``fp4 * block_scale * global_scale``."""
    table = e2m1_table(packed.device, torch.float32)
    values = unpack_fp4(packed, table)
    rows, blocks = values.shape[0], values.shape[1] // BLOCK
    scales = from_blocked(block_scale.to(torch.float32), rows, blocks)
    weight = (values.reshape(rows, blocks, BLOCK) * scales.unsqueeze(-1)).reshape(values.shape)
    weight = weight * float(global_scale)
    # Padded up to a multiple of 16 on the way in, so trim back to the layer's real shape.
    return weight[:out_features, :in_features].to(dtype)


class NVFP4Linear(nn.Module):
    """A ``nn.Linear`` whose weight lives packed and is unpacked inside the forward.

    Holding the unpacked weight would defeat the point: the packed tensors are the whole reason the
    encoder fits. The transient bf16 copy is one layer's worth (~262 MB at the widest) and is freed
    on the way out.
    """

    #: Annotated because `register_buffer` is untyped, so strict mode reads these as `Module`.
    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor
    pre_quant_scale: torch.Tensor | None

    def __init__(
        self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype
        packed_cols = ((in_features + 15) // 16 * 16) // 2
        padded_rows = (out_features + 15) // 16 * 16
        self.register_buffer(
            "weight", torch.empty(padded_rows, packed_cols, dtype=torch.uint8), persistent=True
        )
        # The swizzle pads to 128 rows by 4 blocks, so the scales are stored larger than the weight
        # they describe; sizing this from the weight's own shape truncates every small layer.
        scale_rows = (padded_rows + 127) // 128 * 128
        scale_cols = ((packed_cols * 2 // BLOCK) + 3) // 4 * 4
        self.register_buffer(
            "weight_scale",
            torch.empty(scale_rows, scale_cols, dtype=torch.float8_e4m3fn),
            persistent=True,
        )
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32))
        # AWQ smoothing, on the layers that carry it. Where it is absent the factor is folded into
        # the preceding norm in the same checkpoint, so there is nothing to undo.
        self.register_buffer("pre_quant_scale", None, persistent=True)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.pre_quant_scale
        if smooth is not None:
            x = x * smooth.to(x.dtype)
        weight = dequantize(
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            out_features=self.out_features,
            in_features=self.in_features,
            dtype=x.dtype,
        )
        return torch.nn.functional.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"


def quantize_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ComfyUI's quantiser, for tests only: it is the definition the loader has to invert."""
    global_scale = weight.abs().amax() / (F8_E4M3_MAX * E2M1_MAX)
    blocks = weight.reshape(weight.shape[0], -1, BLOCK)
    block_scale = torch.clamp(
        blocks.abs().amax(dim=-1) / E2M1_MAX / global_scale, max=F8_E4M3_MAX
    ).to(torch.float8_e4m3fn)
    normalised = blocks / (global_scale * block_scale.to(torch.float32)).unsqueeze(-1)
    table = e2m1_table(weight.device, torch.float32)
    flat = normalised.reshape(weight.shape).nan_to_num()
    codes = (flat.unsqueeze(-1) - table).abs().argmin(dim=-1).to(torch.uint8)
    packed = (codes[..., 0::2] << 4) | codes[..., 1::2]
    return packed, to_blocked(block_scale), global_scale
