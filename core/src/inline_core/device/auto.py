"""A minimal default policy: whole graph on the best device, fp16 on cuda, fp32 elsewhere, SDPA.

This is the always-works path (PLAN.md). Phase 3 replaces it with real budgets, offload, profiles.
"""

from __future__ import annotations

from .detect import available_device
from .policy import AttentionBackend, DevicePolicy, Placement, Profile
from .types import Device, DeviceKind, DType

_PROFILE = {
    DeviceKind.CUDA: Profile.GPU_MAX,
    DeviceKind.MPS: Profile.GPU_MAX,
    DeviceKind.CPU: Profile.CPU,
}


class AutoDevicePolicy(DevicePolicy):
    def __init__(self, device: Device | None = None) -> None:
        self._device = device or available_device()

    @property
    def profile(self) -> Profile:
        return _PROFILE[self._device.kind]

    def placement(self, role: str) -> Placement:
        dtype = DType.FP16 if self._device.kind is DeviceKind.CUDA else DType.FP32
        return Placement(self._device, dtype)

    def attention_backend(self) -> AttentionBackend:
        return AttentionBackend.SDPA

    def vae_tiling(self) -> bool:
        return self._device.kind is DeviceKind.CPU
