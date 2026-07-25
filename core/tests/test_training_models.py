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
    on to keep the base full precision rather than dropping it to NF4. A quantized base already
    fits, so offload stays off there no matter the preference."""
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
    # A quantized base already fits, so offload would only add PCIe traffic - never on.
    assert off("on", Quantization.NF4, str(root), archs.KREA2, "raw", 1024) is False


def test_zimage_has_no_four_bit_path_and_says_so(tmp_path) -> None:
    from inline_core.device.policy import Quantization

    # Silently ignoring the request would look like it worked and then OOM anyway.
    with pytest.raises(RuntimeError, match="no 4-bit training path"):
        models.resolve_quant("nf4", str(tmp_path), archs.Z_IMAGE, "deturbo", 512)
    assert models.resolve_quant("auto", str(tmp_path), archs.Z_IMAGE, "deturbo", 1024) is (
        Quantization.NONE
    )
