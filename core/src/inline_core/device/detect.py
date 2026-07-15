"""Best-available device detection. Torch is imported lazily so the core imports without it."""

from __future__ import annotations

from .types import Device, DeviceKind


def available_device() -> Device:
    """The single best device: cuda:0, else mps, else cpu."""
    return available_devices()[0]


def available_devices() -> tuple[Device, ...]:
    """Every usable device: all CUDA GPUs, else one MPS, else one CPU."""
    try:
        import torch
    except ModuleNotFoundError:
        return (Device(DeviceKind.CPU),)
    if torch.cuda.is_available():
        return tuple(Device(DeviceKind.CUDA, i) for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        return (Device(DeviceKind.MPS),)
    return (Device(DeviceKind.CPU),)


def has_nvlink() -> bool:
    """Best-effort: True only when NVLink is confirmed between two GPUs. Conservative (False) on any
    uncertainty, so the policy defaults to the PCIe-friendly PipeFusion split."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() < 2:
                return False
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return bool(pynvml.nvmlDeviceGetNvLinkState(handle, 0))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False
