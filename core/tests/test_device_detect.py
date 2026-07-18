"""Device detection diagnostics — notably the CPU-only-torch warning.

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
    """A CUDA build with no visible GPU is a driver/permissions problem, not a bad wheel — this
    warning would be actively misleading there."""
    monkeypatch.setattr(detect, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=False, cuda_version="12.4")
    )
    assert detect.cpu_only_torch_warning() is None


def test_silent_on_a_machine_with_no_nvidia_gpu(monkeypatch) -> None:
    """A CPU-only wheel is the *correct* choice on an AMD/Apple/CPU box — do not cry wolf."""
    monkeypatch.setattr(detect, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=False, cuda_version=None)
    )
    assert detect.cpu_only_torch_warning() is None


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
