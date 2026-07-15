"""Device and dtype abstractions. Kept independent of torch so nothing load-bearing needs it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeviceKind(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


@dataclass(frozen=True)
class Device:
    kind: DeviceKind
    index: int = 0

    def __str__(self) -> str:
        if self.kind is DeviceKind.CPU:
            return self.kind.value
        return f"{self.kind.value}:{self.index}"
