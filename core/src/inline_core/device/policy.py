"""The device and memory policy interface. Components never self-assign a device; they ask here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .types import Device, DType


class Profile(str, Enum):
    GPU_MAX = "gpu-max"
    LOWVRAM = "lowvram"
    CPU = "cpu"


class AttentionBackend(str, Enum):
    FLASH = "flash"
    XFORMERS = "xformers"
    SDPA = "sdpa"


class OffloadMode(str, Enum):
    """How a component's weights are split between GPU VRAM and CPU RAM.

    NONE       weights stay resident on the GPU (the default — prefer the GPU we have).
    MODEL      diffusers ``enable_model_cpu_offload``: only the *active* component (text encoder,
               then transformer, then VAE) sits on the GPU; the rest waits in CPU RAM. Peak VRAM ≈
               the largest single component. Fast — the standard low-VRAM fit.
    SEQUENTIAL diffusers ``enable_sequential_cpu_offload``: submodules stream on/off the GPU
               layer-by-layer. Lowest peak VRAM, slowest — for GPUs too small for MODEL offload.
    """

    NONE = "none"
    MODEL = "model"
    SEQUENTIAL = "sequential"


class Quantization(str, Enum):
    NONE = "none"
    INT8 = "int8"  # torch-native weight-only, portable
    NF4 = "nf4"  # bitsandbytes, cuda-only


@dataclass(frozen=True)
class Parallel:
    """How the denoiser is split across GPUs (xDiT degrees). Product of degrees = world size (GPUs).
    PipeFusion is PCIe-friendly; Ulysses/Ring want NVLink; CFG applies only to guided models."""

    pipefusion: int = 1
    ulysses: int = 1
    ring: int = 1
    cfg: int = 1
    tensor: int = 1

    @property
    def world_size(self) -> int:
        return self.pipefusion * self.ulysses * self.ring * self.cfg * self.tensor


@dataclass(frozen=True)
class Placement:
    """Where and how a component runs. Chosen by the policy, never by the component."""

    device: Device
    dtype: DType
    offload_mode: OffloadMode = OffloadMode.NONE
    # Multi-GPU (denoiser only): the device group + how it is split. Empty/None = single device.
    devices: tuple[Device, ...] = ()
    parallel: Parallel | None = None

    @property
    def offload(self) -> bool:
        """Whether any CPU offload is in effect (either MODEL or SEQUENTIAL). Kept as a bool so the
        many call sites that only care "is this streaming to CPU?" stay simple."""
        return self.offload_mode is not OffloadMode.NONE


class DevicePolicy(ABC):
    """Owns dtype, device, offload, attention backend, and tiling for a worker."""

    @property
    @abstractmethod
    def profile(self) -> Profile: ...

    @abstractmethod
    def placement(self, role: str) -> Placement:
        """Placement for a component role: text_encoder, denoiser, vae, and so on."""

    @abstractmethod
    def attention_backend(self) -> AttentionBackend: ...

    @abstractmethod
    def vae_tiling(self) -> bool:
        """Whether to tile VAE decode to cap peak memory."""

    def attention_slicing(self) -> bool:
        """Whether to slice attention to cap peak memory. Default off."""
        return False

    def quantization(self) -> Quantization:
        """Weight quantization to fit low memory. Default none."""
        return Quantization.NONE
