"""Best-available device detection. Torch is imported lazily so the core imports without it."""

from __future__ import annotations

from collections.abc import Iterable
from string import ascii_letters

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
    """True when the GPU has native bf16 acceleration.

    NVIDIA: Ampere or newer (compute capability >= 8.0). Turing/Volta (T4, V100) can run bf16 but
    only through a slow unaccelerated path, so the policy prefers fp16 there: same memory footprint,
    but it uses the fp16 tensor cores.

    AMD/ROCm: the compute-capability rule does NOT transfer. HIP reports the gfx arch through the
    same call (gfx1030 -> (10, 3), gfx942 -> (9, 4)), and every AMD major is >= 8, so the NVIDIA
    rule would wave through RDNA2 parts that have no native bf16 at all. Ask torch directly there,
    which is the only vendor-neutral answer available.

    Unknown / non-CUDA falls back to True so bf16 stays the default on anything we can't measure.
    """
    if device.kind is not DeviceKind.CUDA:
        return True
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return bool(torch.cuda.is_bf16_supported())
        major, _ = torch.cuda.get_device_capability(device.index)
        return major >= 8
    except Exception:
        return True


def cpu_only_torch_warning() -> str | None:
    """A warning when the installed torch cannot see an NVIDIA GPU that is physically present.

    PyTorch's default **Windows** wheels on PyPI are CPU-only (the Linux ones bundle CUDA), so a
    plain ``pip install torch`` on Windows yields a *working* install that silently generates on the
    CPU - no error, just ~100x slower. That is the worst kind of failure, so detect it and say so.

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
        # A CUDA build that just can't see a GPU - a driver/permissions issue, not a bad wheel.
        return None
    if not _nvidia_gpu_present():
        return None  # CPU-only wheel on a machine with no NVIDIA GPU: correct, not a problem.
    return (
        "This install has a CPU-ONLY build of PyTorch, but an NVIDIA GPU was detected. "
        "Generation will run on the CPU and be roughly 100x slower. Reinstall torch from the CUDA "
        "index, e.g. `pip install --force-reinstall --extra-index-url "
        "https://download.pytorch.org/whl/cu126 torch` (or re-run `webui.sh --install`)."
    )


def _parse_arch(arch: str) -> tuple[int, int] | None:
    """``sm_120`` to ``(12, 0)``.

    The minor is the **last digit** and the major is everything before it, so ``sm_100`` is (10, 0).
    Reading left to right instead gives (1, 20) and puts the bug precisely on the Blackwell parts.
    Tuned variants (``sm_90a``) drop their trailing letters.
    """
    digits = arch[3:].rstrip(ascii_letters)
    if len(digits) < 2 or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


def arch_list_covers(arches: Iterable[str], major: int, minor: int) -> bool:
    """Whether a wheel's arch list has kernels that will run on a ``(major, minor)`` device.

    CUDA binary compatibility runs **upward within one major only**: an ``sm_8x`` cubin executes on
    any ``sm_8y`` where ``y >= x``. So ``sm_86`` covers an sm_89 Ada card, which is why exact
    matching told every RTX 40-series owner their install was broken. Crossing a major never works,
    so ``sm_90`` does not cover sm_120 and the Blackwell warning stays correct.

    PTX entries (``compute_90``) JIT forward but are not matched here: they do not start with
    ``sm_``, so a PTX-carrying wheel still warns. Rare, and warning is the safe way to be wrong.
    """
    for arch in arches:
        parsed = _parse_arch(arch)
        if parsed is not None and parsed[0] == major and parsed[1] <= minor:
            return True
    return False


def unsupported_arch_warning() -> str | None:
    """A warning when the installed torch has no kernels for the GPU it is about to run on.

    An RTX 50-series card (sm_120) under a wheel built for sm_50..sm_90 is the sharpest case: torch
    reports a CUDA build and a visible device, so every check in ``cpu_only_torch_warning`` passes,
    and the only clue the user gets is PyTorch's own late UserWarning followed by kernels that
    cannot launch. Returns None on any uncertainty, same as the rest of this module.
    """
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return None  # the sm_ vocabulary is NVIDIA's; HIP reports gfx arches through it
        get_arch_list = getattr(torch.cuda, "get_arch_list", None)
        get_capability = getattr(torch.cuda, "get_device_capability", None)
        device_count = getattr(torch.cuda, "device_count", None)
        if get_arch_list is None or get_capability is None or device_count is None:
            return None
        if device_count() < 1:
            return None
        arches = [str(arch) for arch in get_arch_list() if str(arch).startswith("sm_")]
        if not arches:
            return None  # a CPU-only build; cpu_only_torch_warning owns that case
        major, minor = get_capability(0)
        target = f"sm_{major}{minor}"
        if arch_list_covers(arches, major, minor):
            return None
        name = _device_name(torch) or "The detected NVIDIA GPU"
        return (
            f"{name} is compute capability {target}, but this install's PyTorch only has kernels "
            f"for {' '.join(arches)}. Generation will fail or fall back to the CPU. Re-run "
            "`webui.sh --install` (Windows: `.\\webui.bat --install`) to pick the wheel index that "
            "matches the card, or force one with `--torch-index cu130`. On a Blackwell card whose "
            "driver predates R580, use `cu128`, which still has sm_120 but is frozen at torch 2.11."
        )
    except Exception:  # noqa: BLE001 - a diagnostic must never break startup
        return None


def _device_name(torch: object) -> str | None:
    try:
        get_name = getattr(getattr(torch, "cuda", None), "get_device_name", None)
        return str(get_name(0)) if get_name is not None else None
    except Exception:  # noqa: BLE001
        return None


def _nvidia_gpu_present() -> bool:
    """Best-effort check for NVIDIA hardware that does NOT rely on torch's CUDA support - that is
    the whole point, since we are called precisely when torch cannot see the GPU.

    Uses the driver's own `nvidia-smi` rather than NVML: pynvml ships only in the `parallel` extra,
    so it is absent in most installs, whereas nvidia-smi comes with the driver itself. False on any
    uncertainty - a missed warning is better than a false alarm.
    """
    try:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return False
        done = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10, text=True)
        return done.returncode == 0 and "GPU" in (done.stdout or "")
    except Exception:  # noqa: BLE001 - a diagnostic must never break startup
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
