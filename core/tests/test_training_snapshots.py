"""Mid-training snapshots: what gets written, when, and in what format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from inline_core.studio import training_store as ts  # noqa: E402
from inline_core.training import arch as archs  # noqa: E402
from inline_core.training import trainer  # noqa: E402


class _Accelerator:
    is_main_process = True

    def unwrap_model(self, model: Any) -> Any:
        return model


def _arch(export: Any = None) -> archs.TrainingArch:
    base = archs.get("z-image")
    return archs.TrainingArch(
        key=base.key, target_modules=base.target_modules, sigma=base.sigma,
        timestep=base.timestep, target=base.target, forward=base.forward, export_keys=export,
    )


def test_snapshots_are_off_by_default() -> None:
    """A snapshot is a full LoRA, so keeping one per checkpoint costs real disk."""
    assert ts._DEFAULT_HYPERPARAMS["saveSnapshots"] is False


def test_a_snapshot_is_named_for_its_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written: dict[str, Any] = {}

    def fake_save(model: Any, path: str, *, alpha: int, arch: Any) -> None:
        written["path"] = path
        written["alpha"] = alpha
        Path(path).write_bytes(b"")

    monkeypatch.setattr(trainer, "_save_lora", fake_save)
    trainer._save_snapshot(_Accelerator(), object(), tmp_path / "snapshots", 750, 16, _arch())

    assert written["path"].endswith("step-000750.safetensors")
    assert written["alpha"] == 16
    assert (tmp_path / "snapshots" / "step-000750.safetensors").exists()


def test_a_snapshot_goes_through_the_exporting_saver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a copy of the resume checkpoint. That one is a raw PEFT state dict with no alpha and the
    port's own key names, so it would load nowhere but here, which was the original bug."""
    seen: dict[str, Any] = {}

    def fake_save(model: Any, path: str, *, alpha: int, arch: Any) -> None:
        seen["arch"] = arch
        Path(path).write_bytes(b"")

    monkeypatch.setattr(trainer, "_save_lora", fake_save)
    exporter = _arch(export=lambda state: state)
    trainer._save_snapshot(_Accelerator(), object(), tmp_path, 100, 8, exporter)

    assert seen["arch"].export_keys is exporter.export_keys
