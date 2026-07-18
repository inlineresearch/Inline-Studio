"""Device detection diagnostics - notably the CPU-only-torch warning.

PyTorch's default Windows wheels on PyPI are CPU-only, so a plain install there silently generates
on the CPU (~100x slower) with no error. `cpu_only_torch_warning` is the backstop that names it.
"""

from __future__ import annotations

import types

from inline_core.device import detect


def _fake_torch(*, cuda_available: bool, cuda_version: str | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        version=types.SimpleNamespace(cuda=cuda_version),
    )


def test_warns_on_a_cpu_only_wheel_with_an_nvidia_gpu(monkeypatch) -> None:
    """The exact Windows failure mode: CPU-only build (`torch.version.cuda is None`) on a box that
    does have an NVIDIA GPU."""
    monkeypatch.setattr(
        detect, "_nvidia_gpu_present", lambda: True
    )
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=False, cuda_version=None)
    )
    warning = detect.cpu_only_torch_warning()
    assert warning is not None
    assert "CPU-ONLY" in warning
    assert "100x slower" in warning
    assert "download.pytorch.org/whl" in warning  # tells them how to fix it


def test_silent_when_cuda_works(monkeypatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=True, cuda_version="12.4")
    )
    assert detect.cpu_only_torch_warning() is None


def test_silent_on_a_cuda_build_that_cannot_see_a_gpu(monkeypatch) -> None:
    """A CUDA build with no visible GPU is a driver/permissions problem, not a bad wheel - this
    warning would be actively misleading there."""
    monkeypatch.setattr(detect, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=False, cuda_version="12.4")
    )
    assert detect.cpu_only_torch_warning() is None


def test_silent_on_a_machine_with_no_nvidia_gpu(monkeypatch) -> None:
    """A CPU-only wheel is the *correct* choice on an AMD/Apple/CPU box - do not cry wolf."""
    monkeypatch.setattr(detect, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=False, cuda_version=None)
    )
    assert detect.cpu_only_torch_warning() is None


def _fake_gpu_torch(
    *, hip: str | None, capability: tuple[int, int], bf16: bool
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        version=types.SimpleNamespace(cuda=None if hip else "12.4", hip=hip),
        cuda=types.SimpleNamespace(
            get_device_capability=lambda i=0: capability,
            is_bf16_supported=lambda: bf16,
        ),
    )


def test_bf16_gate_uses_the_capability_rule_on_nvidia(monkeypatch) -> None:
    """Ampere+ gets bf16; Turing (T4, capability 7.5) must fall back to fp16, which is the whole
    reason the gate exists."""
    from inline_core.device.types import Device, DeviceKind

    dev = Device(DeviceKind.CUDA, 0)
    for capability, expected in (((8, 6), True), ((9, 0), True), ((7, 5), False), ((7, 0), False)):
        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            _fake_gpu_torch(hip=None, capability=capability, bf16=False),
        )
        assert detect.cuda_supports_bf16(dev) is expected, capability


def test_bf16_gate_asks_torch_directly_on_rocm(monkeypatch) -> None:
    """ROCm reports the gfx arch through get_device_capability (gfx1030 -> (10, 3)), and EVERY AMD
    major is >= 8 - so the NVIDIA rule would wave through RDNA2 parts that have no native bf16.
    On HIP the gate must defer to torch.cuda.is_bf16_supported() instead."""
    from inline_core.device.types import Device, DeviceKind

    dev = Device(DeviceKind.CUDA, 0)
    # RDNA2: capability (10, 3) passes `major >= 8`, but the hardware has no bf16.
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_gpu_torch(hip="6.2.0", capability=(10, 3), bf16=False),
    )
    assert detect.cuda_supports_bf16(dev) is False
    # CDNA3 genuinely supports it.
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_gpu_torch(hip="6.2.0", capability=(9, 4), bf16=True),
    )
    assert detect.cuda_supports_bf16(dev) is True


def test_silent_without_torch(monkeypatch) -> None:
    """A hosted-only (fal) install deliberately has no torch; that is not a misconfiguration."""
    import builtins

    real_import = builtins.__import__

    def _no_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    assert detect.cpu_only_torch_warning() is None
