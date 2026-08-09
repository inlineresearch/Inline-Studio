"""Headless trainer CLI: manifest building and flag parsing, without a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inline_core.training import __main__ as entry
from inline_core.training import manifest as mf


def _touch_media(folder: Path, name: str, caption: str | None = None) -> Path:
    path = folder / name
    path.write_bytes(b"not-a-real-image")
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return path


def test_stage_dataset_copies_sidecars_and_trigger(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _touch_media(src, "hero.png", "a knight")
    _touch_media(src, "walk.mp4", "walking left")

    dest = tmp_path / "staged"
    count = mf.stage_dataset(src, dest, trigger="sks", allow_video=True)

    assert count == 2
    assert (dest / "0000.png").is_file()
    assert (dest / "0001.mp4").is_file()
    assert (dest / "0000.txt").read_text(encoding="utf-8") == "sks, a knight"
    assert (dest / "0001.txt").read_text(encoding="utf-8") == "sks, walking left"


def test_stage_dataset_reads_metadata_jsonl_when_sidecars_missing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _touch_media(src, "clip_a.mp4")
    _touch_media(src, "clip_b.mp4")
    (src / "metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"file_name": "clip_a.mp4", "text": "pixel walk cycle"}),
                json.dumps({"file": "clip_b.mp4", "caption": "pixel idle"}),
            ]
        ),
        encoding="utf-8",
    )

    dest = tmp_path / "staged"
    mf.stage_dataset(src, dest, allow_video=True)

    assert (dest / "0000.txt").read_text(encoding="utf-8") == "pixel walk cycle"
    assert (dest / "0001.txt").read_text(encoding="utf-8") == "pixel idle"


def test_stage_dataset_sidecar_wins_over_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _touch_media(src, "a.mp4", "from sidecar")
    (src / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.mp4", "text": "from jsonl"}), encoding="utf-8"
    )

    dest = tmp_path / "staged"
    mf.stage_dataset(src, dest, allow_video=True)
    assert (dest / "0000.txt").read_text(encoding="utf-8") == "from sidecar"


def test_image_arch_skips_clips(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _touch_media(src, "still.jpg", "face")
    _touch_media(src, "motion.mp4", "walk")

    dest = tmp_path / "staged"
    count = mf.stage_dataset(src, dest, allow_video=False)
    assert count == 1
    assert (dest / "0000.jpg").is_file()
    assert not (dest / "0001.mp4").exists()


def test_build_hyperparams_includes_clip_seconds_for_h3() -> None:
    hp = mf.build_hyperparams(arch="minimax-h3", clip_seconds=1.0, steps=100)
    assert hp["arch"] == "minimax-h3"
    assert hp["baseMode"] == "raw"
    assert hp["clipSeconds"] == 1.0
    assert hp["steps"] == 100


def test_build_hyperparams_omits_clip_seconds_for_image_archs() -> None:
    hp = mf.build_hyperparams(arch="z-image", clip_seconds=2.0)
    assert "clipSeconds" not in hp


def test_prepare_run_writes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _touch_media(src, "0000.mp4", "pixel art walk")
    models = tmp_path / "models"
    models.mkdir()
    work = tmp_path / "work"
    out = tmp_path / "out" / "lora.safetensors"

    path, manifest = mf.prepare_run(
        dataset=src,
        models_dir=models,
        output=out,
        work_dir=work,
        arch="minimax-h3",
        clip_seconds=1.0,
        steps=12,
        resolution=512,
        trigger="pxart",
    )

    assert path == work / "manifest.json"
    assert path.is_file()
    assert manifest["arch"] == "minimax-h3"
    assert manifest["hyperparams"]["clipSeconds"] == 1.0
    assert manifest["outputPath"] == str(out.resolve())
    assert Path(manifest["datasetDir"]).is_dir()
    caption = next(Path(manifest["datasetDir"]).glob("*.txt")).read_text(encoding="utf-8")
    assert caption == "pxart, pixel art walk"


def test_cli_dry_run_prints_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _touch_media(src, "a.png", "still")
    models = tmp_path / "models"
    models.mkdir()
    work = tmp_path / "work"

    code = entry.main(
        [
            "--dataset",
            str(src),
            "--models-dir",
            str(models),
            "--work-dir",
            str(work),
            "--arch",
            "minimax-h3",
            "--clip-seconds",
            "1",
            "--steps",
            "8",
            "--dry-run",
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["arch"] == "minimax-h3"
    assert printed["hyperparams"]["clipSeconds"] == 1.0
    assert printed["hyperparams"]["steps"] == 8
    assert (work / "manifest.json").is_file()


def test_cli_manifest_path_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Studio orchestrator path: a single manifest.json argument must not require flags."""
    manifest = {
        "arch": "minimax-h3",
        "baseMode": "raw",
        "hyperparams": {"resolution": 512, "steps": 1},
        "outputPath": str(tmp_path / "out.safetensors"),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    called: dict[str, object] = {}

    def fake_run(m: dict) -> int:
        called["m"] = m
        return 0

    monkeypatch.setattr(entry, "run_training", fake_run)
    code = entry.main([str(path)])
    assert code == 0
    assert called["m"]["arch"] == "minimax-h3"


def test_cli_missing_dataset_errors(capsys: pytest.CaptureFixture[str]) -> None:
    code = entry.main(["--arch", "minimax-h3", "--dry-run"])
    assert code == 2
    err = capsys.readouterr()
    # protocol.error goes to stdout as JSON
    lines = [line for line in err.out.splitlines() if line.startswith("{")]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["type"] == "error"
    assert "dataset" in payload["message"].lower()


def test_cli_empty_args_errors(capsys: pytest.CaptureFixture[str]) -> None:
    code = entry.main([])
    assert code == 2
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["type"] == "error"
