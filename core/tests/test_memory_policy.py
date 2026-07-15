from __future__ import annotations

from inline_core.device.memory import MemoryPolicy
from inline_core.device.policy import Parallel, Profile, Quantization
from inline_core.device.types import Device, DeviceKind

_CUDA = Device(DeviceKind.CUDA, 0)
_CPU = Device(DeviceKind.CPU)


def _cuda(count: int) -> tuple[Device, ...]:
    return tuple(Device(DeviceKind.CUDA, i) for i in range(count))


def test_ample_vram_is_gpu_max() -> None:
    policy = MemoryPolicy(_CUDA, vram_gb=24)
    assert policy.profile is Profile.GPU_MAX
    assert policy.placement("denoiser").offload is False
    assert policy.quantization() is Quantization.NONE
    assert policy.attention_slicing() is False


def test_tight_vram_is_lowvram_but_keeps_weights_on_gpu() -> None:
    # Always prefer the GPU: lowvram saves memory with slicing/tiling/int8, NOT by CPU offload.
    policy = MemoryPolicy(_CUDA, vram_gb=6)
    assert policy.profile is Profile.LOWVRAM
    assert policy.placement("denoiser").offload is False
    assert policy.attention_slicing() is True
    assert policy.vae_tiling() is True
    assert policy.quantization() is Quantization.INT8


def test_lowvram_offload_is_opt_in_via_env(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_ALLOW_CPU_OFFLOAD", "1")
    policy = MemoryPolicy(_CUDA, vram_gb=6)
    assert policy.profile is Profile.LOWVRAM
    assert policy.placement("denoiser").offload is True


def test_lowvram_offload_can_be_forced_off_via_arg() -> None:
    # An explicit allow_offload arg wins over the env default.
    policy = MemoryPolicy(_CUDA, vram_gb=6, allow_offload=True)
    assert policy.placement("denoiser").offload is True


def test_cpu_uses_fp32_and_quantizes_on_low_ram() -> None:
    low = MemoryPolicy(_CPU, ram_gb=16)
    assert low.profile is Profile.CPU
    assert low.placement("denoiser").dtype.value == "fp32"
    assert low.placement("denoiser").offload is False
    assert low.quantization() is Quantization.INT8
    assert low.vae_tiling() is True

    ample = MemoryPolicy(_CPU, ram_gb=128)
    assert ample.quantization() is Quantization.NONE


def test_env_profile_override(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_PROFILE", "lowvram")
    assert MemoryPolicy(_CUDA, vram_gb=48).profile is Profile.LOWVRAM


def test_single_gpu_denoiser_is_not_parallel() -> None:
    assert MemoryPolicy(devices=_cuda(1), vram_gb=24).placement("denoiser").parallel is None


def test_two_gpus_pcie_split_with_pipefusion() -> None:
    placement = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=False).placement("denoiser")
    assert placement.parallel == Parallel(pipefusion=2)
    assert placement.parallel is not None and placement.parallel.world_size == 2
    assert len(placement.devices) == 2


def test_two_gpus_nvlink_split_with_ulysses() -> None:
    placement = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=True).placement("denoiser")
    assert placement.parallel == Parallel(ulysses=2)


def test_non_denoiser_role_stays_single_device() -> None:
    policy = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=False)
    assert policy.placement("vae").parallel is None


def test_env_parallel_override(monkeypatch) -> None:
    monkeypatch.setenv("INLINE_PARALLEL", "pipefusion=2,ulysses=2")
    placement = MemoryPolicy(devices=_cuda(4), vram_gb=24).placement("denoiser")
    assert placement.parallel == Parallel(pipefusion=2, ulysses=2)
