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


def cuda_supports_bf16(device: Device) -> bool:
    """True when the CUDA GPU has native bf16 acceleration — Ampere or newer (compute capability
    >= 8.0). Turing/Volta (T4, V100) can run bf16 but only through a slow unaccelerated path, so the
    policy prefers fp16 there (same memory footprint, but it uses the fp16 tensor cores). Unknown /
    non-CUDA falls back to True so bf16 stays the default on anything we can't measure."""
    if device.kind is not DeviceKind.CUDA:
        return True
    try:
        import torch

        major, _ = torch.cuda.get_device_capability(device.index)
        return major >= 8
    except Exception:
        return True


def cpu_only_torch_warning() -> str | None:
    """A warning when the installed torch cannot see an NVIDIA GPU that is physically present.

    PyTorch's default **Windows** wheels on PyPI are CPU-only (the Linux ones bundle CUDA), so a
    plain ``pip install torch`` on Windows yields a *working* install that silently generates on the
    CPU — no error, just ~100x slower. That is the worst kind of failure, so detect it and say so.

    The tell is ``torch.version.cuda is None`` (a CPU-only build) while an NVIDIA driver is present.
    Returns None when there is nothing to report: no torch at all (a deliberate hosted-only
    install), CUDA working normally, Apple Silicon, or a genuinely GPU-less machine.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None  # hosted-only install; nothing to warn about
    if torch.cuda.is_available():
        return None
    if getattr(torch.version, "cuda", None) is not None:
        # A CUDA build that just can't see a GPU — a driver/permissions issue, not a bad wheel.
        return None
    if not _nvidia_gpu_present():
        return None  # CPU-only wheel on a machine with no NVIDIA GPU: correct, not a problem.
    return (
        "This install has a CPU-ONLY build of PyTorch, but an NVIDIA GPU was detected. "
        "Generation will run on the CPU and be roughly 100x slower. Reinstall torch from the CUDA "
        "index, e.g. `pip install --force-reinstall --extra-index-url "
        "https://download.pytorch.org/whl/cu124 torch` (or re-run `webui.sh --install`)."
    )


def _nvidia_gpu_present() -> bool:
    """Best-effort check for NVIDIA hardware that does NOT rely on torch's CUDA support — that is
    the whole point, since we are called precisely when torch cannot see the GPU.

    Uses the driver's own `nvidia-smi` rather than NVML: pynvml ships only in the `parallel` extra,
    so it is absent in most installs, whereas nvidia-smi comes with the driver itself. False on any
    uncertainty — a missed warning is better than a false alarm.
    """
    try:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return False
        done = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10, text=True)
        return done.returncode == 0 and "GPU" in (done.stdout or "")
    except Exception:  # noqa: BLE001 — a diagnostic must never break startup
        return False


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
