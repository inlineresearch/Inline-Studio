"""Trainer-side component resolution: which base and which training adapter a run picks.

Both architectures keep their de-distillation adapter in the same ``models/loras/`` folder, so
"any file with adapter in the name" is no longer good enough - picking Z-Image's adapter for a
Krea 2 run would fail deep inside the fuse with a wall of unmatched layers.
"""

from __future__ import annotations

import pytest

models = pytest.importorskip("inline_core.training.models")
from inline_core.training import arch as archs  # noqa: E402


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "INLINE_ZIMAGE_TRAIN_ADAPTER", "INLINE_KREA2_TRAIN_ADAPTER",
        "INLINE_ZIMAGE_MODEL", "INLINE_KREA2_MODEL",
        "INLINE_ZIMAGE_VAE", "INLINE_KREA2_VAE",
        "INLINE_ZIMAGE_TEXT_ENCODER", "INLINE_KREA2_TEXT_ENCODER",
    ):
        monkeypatch.delenv(var, raising=False)


def _loras(tmp_path, *names):
    root = tmp_path / "models"
    (root / "loras").mkdir(parents=True)
    for name in names:
        (root / "loras" / name).write_bytes(b"")
    return root


def test_each_arch_picks_its_own_training_adapter(tmp_path) -> None:
    root = _loras(
        tmp_path,
        "krea2_turbo_training_adapter_v1.safetensors",
        "zimage_turbo_training_adapter_v2.safetensors",
    )

    krea2 = models._adapter_path(root, archs.KREA2, "turbo_adapter")
    zimage = models._adapter_path(root, archs.Z_IMAGE, "turbo_adapter")

    assert krea2 is not None and "krea2" in krea2
    assert zimage is not None and "zimage" in zimage


def test_an_undistilled_base_needs_no_adapter(tmp_path) -> None:
    root = _loras(tmp_path)

    # Krea 2 RAW was never distilled, and Z-Image de-turbo trains without one.
    assert models._adapter_path(root, archs.KREA2, "raw") is None
    assert models._adapter_path(root, archs.Z_IMAGE, "deturbo") is None


def test_a_missing_adapter_says_what_to_do(tmp_path) -> None:
    root = _loras(tmp_path, "some_style_lora.safetensors")

    with pytest.raises(RuntimeError, match="training adapter"):
        models._adapter_path(root, archs.KREA2, "turbo_adapter")


def test_an_explicit_adapter_override_wins(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _loras(tmp_path, "krea2_turbo_training_adapter_v1.safetensors")
    chosen = tmp_path / "elsewhere.safetensors"
    chosen.write_bytes(b"")
    monkeypatch.setenv("INLINE_KREA2_TRAIN_ADAPTER", str(chosen))

    assert models._adapter_path(root, archs.KREA2, "turbo_adapter") == str(chosen)


def test_auto_quantization_accounts_for_resolution(monkeypatch, tmp_path) -> None:
    """The bug this pins: a 46GB card holds Krea 2's 26GB base fine, then OOMs at 1024 where the
    activations alone want ~21GB. Weights alone are not enough to decide."""
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    base = root / "diffusion_models" / "krea2_raw_bf16.safetensors"
    base.write_bytes(b"")
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.setenv("INLINE_KREA2_MODEL", str(base))
    monkeypatch.setattr(models, "_base_size", lambda *a: 26 * 1024**3)

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _i: type("P", (), {"total_memory": 46 * 1024**3})(),
    )

    from inline_core.device.policy import Quantization

    resolve = models.resolve_quant
    assert resolve("auto", str(root), archs.KREA2, "raw", 512) is Quantization.NONE
    assert resolve("auto", str(root), archs.KREA2, "raw", 1024) is Quantization.NF4


def test_offload_fits_a_bf16_base_that_would_not_otherwise(monkeypatch, tmp_path) -> None:
    """bf16 1024 on a 45GB card: base (26GB) + activations (~21GB) overflow, so auto-offload turns
    on to keep the base full precision rather than dropping it to NF4. Under a quantized base AUTO
    stays off, but an explicit on/off is the user's answer and wins."""
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / "krea2_raw_bf16.safetensors").write_bytes(b"")
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.setattr(models, "_base_size", lambda *a: 26 * 1024**3)

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _i: type("P", (), {"total_memory": 45 * 1024**3})(),
    )

    from inline_core.device.policy import Quantization

    off = models.resolve_offload
    # bf16 base: auto offloads at 1024 (won't fit), leaves it resident at 512 (fits).
    assert off("auto", Quantization.NONE, str(root), archs.KREA2, "raw", 1024) is True
    assert off("auto", Quantization.NONE, str(root), archs.KREA2, "raw", 512) is False
    assert off("on", Quantization.NONE, str(root), archs.KREA2, "raw", 512) is True
    assert off("off", Quantization.NONE, str(root), archs.KREA2, "raw", 1024) is False
    # Auto stays off under a quantized base; an explicit choice wins.
    assert off("auto", Quantization.NF4, str(root), archs.KREA2, "raw", 1024) is False
    assert off("on", Quantization.NF4, str(root), archs.KREA2, "raw", 1024) is True
    assert off("off", Quantization.NF4, str(root), archs.KREA2, "raw", 1024) is False


def test_zimage_has_no_four_bit_path_and_says_so(tmp_path) -> None:
    from inline_core.device.policy import Quantization

    # Silently ignoring the request would look like it worked and then OOM anyway.
    with pytest.raises(RuntimeError, match="no 4-bit training path"):
        models.resolve_quant("nf4", str(tmp_path), archs.Z_IMAGE, "deturbo", 512)
    assert models.resolve_quant("auto", str(tmp_path), archs.Z_IMAGE, "deturbo", 1024) is (
        Quantization.NONE
    )


# --- mmap preflight -------------------------------------------------------------------------
#
# safetensors maps the whole checkpoint in one call, and vm.overcommit_memory=0 refuses a single
# mapping larger than RAM+swap. The raw kernel error names the checkpoint, so it reads as a corrupt
# download, and it only fires after the precache has already cost twenty minutes.


def _fake_env(monkeypatch, *, mode, ram_gib, swap_gib=0, size_gib=62, ratio=50):
    values = {
        "/proc/sys/vm/overcommit_memory": mode,
        "/proc/sys/vm/overcommit_ratio": ratio,
    }
    monkeypatch.setattr(models, "_proc_int", lambda p: values.get(p))
    monkeypatch.setattr(
        models, "_memory_totals", lambda: (ram_gib * 1024**3, swap_gib * 1024**3)
    )
    monkeypatch.setattr(models, "_base_size", lambda *a: int(size_gib * 1024**3))
    monkeypatch.setattr(models, "_base_file", lambda *a: "/m/minimax_h3_fl2va_bf16.safetensors")


def test_refuses_a_checkpoint_bigger_than_ram_plus_swap(monkeypatch) -> None:
    _fake_env(monkeypatch, mode=0, ram_gib=30, swap_gib=0, size_gib=62)
    with pytest.raises(RuntimeError) as caught:
        models.check_base_mappable("/m", "minimax-h3", "raw")
    message = str(caught.value)
    assert "minimax_h3_fl2va_bf16.safetensors" in message
    # The fix has to be in the message, since this is the whole point of checking early.
    assert "vm.overcommit_memory=1" in message


def test_swap_counts_toward_the_ceiling(monkeypatch) -> None:
    _fake_env(monkeypatch, mode=0, ram_gib=30, swap_gib=64, size_gib=62)
    models.check_base_mappable("/m", "minimax-h3", "raw")


def test_overcommit_always_is_never_refused(monkeypatch) -> None:
    """Mode 1 lets any mapping through, however large."""
    _fake_env(monkeypatch, mode=1, ram_gib=8, swap_gib=0, size_gib=62)
    models.check_base_mappable("/m", "minimax-h3", "raw")


def test_strict_mode_uses_the_overcommit_ratio(monkeypatch) -> None:
    """Mode 2 allows only ratio% of RAM plus swap, so 50% of 200GB does not fit 62GB... it does,
    but 50% of 100GB with no swap does not."""
    _fake_env(monkeypatch, mode=2, ram_gib=100, swap_gib=0, size_gib=62, ratio=50)
    with pytest.raises(RuntimeError):
        models.check_base_mappable("/m", "minimax-h3", "raw")
    _fake_env(monkeypatch, mode=2, ram_gib=100, swap_gib=0, size_gib=40, ratio=50)
    models.check_base_mappable("/m", "minimax-h3", "raw")


def test_fails_open_when_the_machine_cannot_be_read(monkeypatch) -> None:
    """No /proc (not Linux) or an unmeasurable checkpoint must never block a run that would work."""
    monkeypatch.setattr(models, "_proc_int", lambda _p: None)
    models.check_base_mappable("/m", "minimax-h3", "raw")

    _fake_env(monkeypatch, mode=0, ram_gib=30, size_gib=62)
    monkeypatch.setattr(models, "_memory_totals", lambda: (0, 0))
    models.check_base_mappable("/m", "minimax-h3", "raw")

    _fake_env(monkeypatch, mode=0, ram_gib=30, size_gib=62)
    monkeypatch.setattr(models, "_base_size", lambda *a: 0)
    models.check_base_mappable("/m", "minimax-h3", "raw")


def test_an_explicit_offload_choice_beats_the_quant_heuristic(monkeypatch, tmp_path) -> None:
    """Dead for H3, always 4-bit, whose clip activations are what overflow the card."""
    from inline_core.device.policy import Quantization

    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setattr(models, "_base_size", lambda *a: 12 * 1024**3)
    off = models.resolve_offload

    assert off("on", Quantization.NF4, str(root), archs.MINIMAX_H3, "raw", 512) is True
    assert off("off", Quantization.NF4, str(root), archs.MINIMAX_H3, "raw", 512) is False
    assert off("auto", Quantization.NF4, str(root), archs.MINIMAX_H3, "raw", 512) is False


def test_an_unknown_offload_preference_still_raises(monkeypatch, tmp_path) -> None:
    """The reorder must not let a typo fall through to the auto path and silently mean 'off'."""
    from inline_core.device.policy import Quantization

    monkeypatch.setattr(models, "_base_size", lambda *a: 12 * 1024**3)
    with pytest.raises(RuntimeError):
        models.resolve_offload("yes", Quantization.NF4, str(tmp_path), archs.MINIMAX_H3, "raw", 512)
