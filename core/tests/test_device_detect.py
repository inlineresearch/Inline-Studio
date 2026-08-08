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
    caps = (((12, 0), True), ((8, 6), True), ((9, 0), True), ((7, 5), False), ((7, 0), False))
    for capability, expected in caps:
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


# What a cu124 wheel reports - the build every Windows install used to be pinned to.
_CU124_ARCHES = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def _fake_arch_torch(
    *, capability: tuple[int, int], arches: list[str], name: str = "NVIDIA GeForce RTX 5070 Ti"
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        version=types.SimpleNamespace(cuda="12.4", hip=None),
        cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_arch_list=lambda: arches,
            get_device_capability=lambda i=0: capability,
            get_device_name=lambda i=0: name,
        ),
    )


def test_warns_when_the_wheel_has_no_kernels_for_the_card(monkeypatch) -> None:
    """The RTX 50-series failure: torch is a CUDA build and the device is visible, so every check in
    cpu_only_torch_warning passes and the user is left with PyTorch's own cryptic UserWarning."""
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(12, 0), arches=_CU124_ARCHES),
    )
    warning = detect.unsupported_arch_warning()
    assert warning is not None
    assert "sm_120" in warning
    assert "RTX 5070 Ti" in warning
    assert "--torch-index" in warning  # tells them how to fix it


def test_silent_when_the_wheel_covers_the_card(monkeypatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(8, 6), arches=_CU124_ARCHES),
    )
    assert detect.unsupported_arch_warning() is None


def test_arch_warning_accepts_a_tuned_variant(monkeypatch) -> None:
    """Wheels list per-architecture variants like sm_90a; that is still a match for sm_90."""
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(9, 0), arches=["sm_80", "sm_90a"]),
    )
    assert detect.unsupported_arch_warning() is None


def test_arch_warning_silent_when_torch_cannot_answer(monkeypatch) -> None:
    """An older torch has no get_arch_list, and a CPU-only build reports no sm_ arches at all -
    cpu_only_torch_warning owns that second case. Neither may produce a false alarm here."""
    monkeypatch.setitem(
        __import__("sys").modules, "torch", _fake_torch(cuda_available=False, cuda_version="12.4")
    )
    assert detect.unsupported_arch_warning() is None
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(12, 0), arches=[]),
    )
    assert detect.unsupported_arch_warning() is None


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


# --- CUDA within-major binary compatibility -----------------------------------------------------
#
# A cubin built for sm_8x runs on any sm_8y where y >= x, so sm_86 covers an sm_89 Ada card. Exact
# per-minor matching told every RTX 40-series owner their install was broken.

_CU130_ARCHES = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_arch_parse_reads_the_minor_as_the_last_digit() -> None:
    """sm_120 is (12, 0), not (1, 20). Left-to-right puts the bug on the Blackwell parts."""
    assert detect._parse_arch("sm_120") == (12, 0)
    assert detect._parse_arch("sm_100") == (10, 0)
    assert detect._parse_arch("sm_90a") == (9, 0)  # tuned variant
    assert detect._parse_arch("compute_90") is None


def test_ada_is_covered_by_ampere_kernels(monkeypatch) -> None:
    """The RTX 4080 false positive: sm_89 against a wheel whose newest 8.x is sm_86."""
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(8, 9), arches=_CU124_ARCHES),
    )
    assert detect.unsupported_arch_warning() is None


def test_compatibility_runs_upward_only(monkeypatch) -> None:
    """An sm_86 cubin does NOT run on an sm_80 A100, so that must still warn."""
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(8, 0), arches=["sm_86"]),
    )
    assert detect.unsupported_arch_warning() is not None


def test_a_dropped_architecture_still_warns(monkeypatch) -> None:
    """cu130 dropped Volta; sm_70 has no same-major kernel at or below it."""
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        _fake_arch_torch(capability=(7, 0), arches=_CU130_ARCHES),
    )
    assert detect.unsupported_arch_warning() is not None


def test_blackwell_majors_do_not_cover_each_other() -> None:
    """sm_100 and sm_120 are both Blackwell but different majors, so neither covers the other."""
    assert detect.arch_list_covers(["sm_100"], 12, 0) is False
    assert detect.arch_list_covers(["sm_120"], 10, 0) is False
    assert detect.arch_list_covers(_CU130_ARCHES, 10, 0) is True
    assert detect.arch_list_covers(_CU130_ARCHES, 12, 0) is True


# --- the install-time probe --------------------------------------------------------------------


def _fake_probe_torch(
    *, version: str, arches: list[str], capability: tuple[int, int] | None = (12, 0),
    hip: str | None = None, devices: int = 1,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__=version,
        version=types.SimpleNamespace(cuda=None if "+cpu" in version else "12.8", hip=hip),
        cuda=types.SimpleNamespace(
            device_count=lambda: devices,
            get_arch_list=lambda: arches,
            get_device_capability=lambda i=0: capability,
        ),
    )


def _probe_with(monkeypatch, torch_stub) -> dict:
    from inline_core.device import probe as probe_mod

    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    return probe_mod.probe()


def test_probe_reports_an_uncovered_wheel_as_replaceable(monkeypatch) -> None:
    """The 5060 Ti case: a cu126 wheel on sm_120, and the +cuXXX tag makes it safe to replace."""
    got = _probe_with(
        monkeypatch, _fake_probe_torch(version="2.9.0+cu126", arches=_CU124_ARCHES)
    )
    assert got["status"] == "uncovered"
    assert got["replaceable"] is True


def test_probe_reports_ada_as_covered(monkeypatch) -> None:
    got = _probe_with(
        monkeypatch,
        _fake_probe_torch(version="2.13.0+cu130", arches=_CU130_ARCHES, capability=(8, 9)),
    )
    assert got["status"] == "covered"


def test_probe_never_offers_to_replace_a_rocm_build(monkeypatch) -> None:
    """The safety gate. A ROCm build fails the sm_ rule too, and reinstalling over someone's
    deliberate choice is worse than the wrong wheel."""
    got = _probe_with(
        monkeypatch,
        _fake_probe_torch(version="2.9.0+rocm6.2", arches=[], hip="6.2.0"),
    )
    assert got["status"] == "rocm"
    assert got["replaceable"] is False


def test_probe_never_offers_to_replace_an_untagged_build(monkeypatch) -> None:
    """A nightly or hand-built wheel carries no +cpu/+cuXXX tag, so leave it alone."""
    got = _probe_with(
        monkeypatch, _fake_probe_torch(version="2.14.0.dev20260101", arches=_CU124_ARCHES)
    )
    assert got["status"] == "uncovered"
    assert got["replaceable"] is False


def test_probe_survives_a_broken_torch(monkeypatch) -> None:
    """Never covered on uncertainty: a torch that raises must read as unknown."""
    class Boom:
        __version__ = "2.13.0+cu130"

        def __getattr__(self, _name: str):
            raise RuntimeError("broken install")

    got = _probe_with(monkeypatch, Boom())
    assert got["status"] == "unknown"
    assert got["status"] != "covered"
