"""The memory-aware device policy: measure RAM/VRAM, pick a profile, own dtype/offload/quant/tiling.

Profiles: gpu-max (ample VRAM), lowvram (tight VRAM), cpu (fp32, tiling, int8 to fit RAM). We always
prefer the GPU: even under lowvram, weights stay resident on the GPU (tiling + attention slicing +
int8 do the memory saving) and we do NOT auto-offload to CPU — offloading is slow and defeats "use
the GPU we have". Set INLINE_ALLOW_CPU_OFFLOAD=1 to opt back into streaming modules on/off the GPU
for extreme cases. Override the profile/budget with INLINE_PROFILE and INLINE_VRAM_BUDGET_GB.
Detection is lazy so the core imports without torch or psutil; an unavailable measurement keeps the
policy conservative.
"""

from __future__ import annotations

import os

from .detect import available_devices, has_nvlink
from .policy import AttentionBackend, DevicePolicy, Parallel, Placement, Profile, Quantization
from .types import Device, DeviceKind, DType

_GPU_MAX_MIN_VRAM_GB = 16.0  # at or above -> gpu-max, else lowvram
_QUANT_VRAM_GB = 10.0  # lowvram below this -> int8
_QUANT_RAM_GB = 48.0  # cpu below this -> int8


def _system_ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except ModuleNotFoundError:
        return None


def _vram_gb(device: Device) -> float | None:
    if device.kind is not DeviceKind.CUDA:
        return None
    try:
        import torch

        return torch.cuda.mem_get_info(device.index)[1] / 1e9
    except Exception:
        return None


def _env_profile() -> Profile | None:
    value = os.environ.get("INLINE_PROFILE", "").strip().lower()
    return next((p for p in Profile if p.value == value), None)


def _env_budget() -> float | None:
    value = os.environ.get("INLINE_VRAM_BUDGET_GB", "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _env_allow_offload() -> bool:
    """Opt back into CPU offload under lowvram. Off by default: we keep weights on the GPU."""
    value = os.environ.get("INLINE_ALLOW_CPU_OFFLOAD", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _env_parallel() -> Parallel | None:
    """Parse INLINE_PARALLEL like `pipefusion=2,ulysses=2` into a degree spec."""
    value = os.environ.get("INLINE_PARALLEL", "").strip()
    if not value:
        return None
    valid = {"pipefusion", "ulysses", "ring", "cfg", "tensor"}
    degrees: dict[str, int] = {}
    for part in value.split(","):
        key, _, raw = part.partition("=")
        if key.strip() in valid:
            try:
                degrees[key.strip()] = int(raw)
            except ValueError:
                continue
    return Parallel(**degrees) if degrees else None


class MemoryPolicy(DevicePolicy):
    def __init__(
        self,
        device: Device | None = None,
        *,
        ram_gb: float | None = None,
        vram_gb: float | None = None,
        profile: Profile | None = None,
        devices: tuple[Device, ...] | None = None,
        nvlink: bool | None = None,
        parallel: Parallel | None = None,
        allow_offload: bool | None = None,
    ) -> None:
        self._devices = devices if devices is not None else available_devices()
        self._device = device or self._devices[0]
        self._ram_gb = ram_gb if ram_gb is not None else _system_ram_gb()
        self._vram_gb = vram_gb if vram_gb is not None else _vram_gb(self._device)
        self._profile = profile or _env_profile() or self._choose_profile()
        self._nvlink = nvlink if nvlink is not None else has_nvlink()
        self._parallel = parallel if parallel is not None else _env_parallel()
        self._allow_offload = allow_offload if allow_offload is not None else _env_allow_offload()

    def _choose_profile(self) -> Profile:
        if self._device.kind is DeviceKind.CPU:
            return Profile.CPU
        if self._device.kind is DeviceKind.MPS:
            return Profile.GPU_MAX  # unified memory; offload semantics differ
        budget = _env_budget() or self._vram_gb
        if budget is not None and budget < _GPU_MAX_MIN_VRAM_GB:
            return Profile.LOWVRAM
        return Profile.GPU_MAX

    @property
    def profile(self) -> Profile:
        return self._profile

    def placement(self, role: str) -> Placement:
        if self._profile is Profile.CPU:
            return Placement(self._device, DType.FP32)
        # Always prefer the GPU: even under lowvram, keep weights resident (tiling/slicing/int8 save
        # memory instead). Only stream modules to CPU when explicitly opted in via env.
        offload = self._profile is Profile.LOWVRAM and self._allow_offload
        if role == "denoiser":
            parallel = self._denoiser_parallel()
            if parallel is not None:
                cuda = tuple(d for d in self._devices if d.kind is DeviceKind.CUDA)
                return Placement(
                    self._device,
                    DType.BF16,
                    offload=offload,
                    devices=cuda[: parallel.world_size],
                    parallel=parallel,
                )
        return Placement(self._device, DType.BF16, offload=offload)

    def _denoiser_parallel(self) -> Parallel | None:
        """Split the denoiser across GPUs when there are 2+. An explicit override wins; else auto:
        PipeFusion on PCIe (no NVLink), sequence-parallel (Ulysses) on NVLink."""
        if self._parallel is not None:
            return self._parallel if self._parallel.world_size > 1 else None
        cuda = [d for d in self._devices if d.kind is DeviceKind.CUDA]
        if len(cuda) < 2:
            return None
        return Parallel(ulysses=len(cuda)) if self._nvlink else Parallel(pipefusion=len(cuda))

    def attention_backend(self) -> AttentionBackend:
        return AttentionBackend.SDPA

    def vae_tiling(self) -> bool:
        return self._profile in (Profile.LOWVRAM, Profile.CPU)

    def attention_slicing(self) -> bool:
        return self._profile in (Profile.LOWVRAM, Profile.CPU)

    def quantization(self) -> Quantization:
        if self._profile is Profile.LOWVRAM and _below(self._vram_gb, _QUANT_VRAM_GB):
            return Quantization.INT8
        if self._profile is Profile.CPU and _below(self._ram_gb, _QUANT_RAM_GB):
            return Quantization.INT8
        return Quantization.NONE


def _below(measured: float | None, threshold: float) -> bool:
    return measured is not None and measured < threshold
