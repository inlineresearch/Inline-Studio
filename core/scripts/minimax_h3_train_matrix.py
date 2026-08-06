"""VRAM + step-time sweep for MiniMax H3 LoRA training: the cells behind the README benchmark table.

    cd core && PYTHONPATH=src .venv/bin/python scripts/minimax_h3_train_matrix.py --dataset <dir>

One cell = one real run of `python -m inline_core.training`, the same entry point the Trainer tab
spawns, so a number here is a number a user would see.

Held fixed at the settings the other architectures' rows used: 12 steps, rank 16, batch 1, gradient
checkpointing on. Only resolution varies, because H3 has a single base mode and is 4-bit only: its
base is 40 GB after the AdaLN factorisation, so a bf16 cell would be measuring an OOM.

The peak here is not the peak a user waits on. H3 encodes latents and captions in two passes that
never overlap the base, and the conditioner pass is the tallest of the three, so the run's
high-water mark is set before training starts. Both are reported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_CORE = _REPO / "core"
_DEFAULT_OUT = _REPO / "outputs" / "minimax-h3-bench" / "train"

#: Resolutions to sweep. 512 is the practical setting; 768 matches H3's own short edge at inference.
CELLS: tuple[int, ...] = (512, 768)

_STEPS = 12
_RANK = 16


def _manifest(
    work: Path, dataset: Path, models: Path, resolution: int, steps: int = _STEPS
) -> Path:
    """The same manifest shape `studio/training.py::_prepare` writes."""
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runId": work.name,
        "workingDir": str(work),
        "datasetDir": str(dataset),
        "checkpointDir": str(checkpoints),
        "outputPath": str(work / "lora.safetensors"),
        "resumeFrom": None,
        "modelsDir": str(models),
        "arch": "minimax-h3",
        # H3 ships one undistilled build per partition; `raw` is that mode's key.
        "baseMode": "raw",
        "triggerWord": "",
        "hyperparams": {
            "arch": "minimax-h3",
            "baseMode": "raw",
            "baseQuant": "auto",
            "offload": "off",
            "loraScope": "full",
            "captionDropout": 0.0,
            "flipAugment": False,
            "rank": _RANK,
            "alpha": _RANK,
            "learningRate": 1e-4,
            "batchSize": 1,
            "steps": steps,
            "saveEvery": steps,
            "resolution": resolution,
        },
        "gpuIds": [],
    }
    path = work / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _run_cell(python: str, manifest: Path, log: Path) -> dict[str, object]:
    """Drain the JSON-line protocol, keeping the VRAM readings and the step timing.

    The precache peak is read from the progress lines before the first training step; the training
    peak is the last reading. They are different numbers on H3 and conflating them would overstate
    what training itself costs.
    """
    env = {**os.environ, "PYTHONPATH": str(_CORE / "src")}
    proc = subprocess.Popen(
        [python, "-m", "inline_core.training", str(manifest)],
        cwd=str(_CORE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    vram: float | None = None
    error: str | None = None
    first_step_at: float | None = None
    last_step_at: float | None = None
    steps_seen = 0
    losses: list[float] = []
    started = time.perf_counter()
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = message.get("type")
        if kind == "progress":
            if message.get("vram") is not None:
                vram = float(message["vram"])
            if message.get("loss") is not None:
                losses.append(float(message["loss"]))
            if message.get("step"):
                steps_seen = int(message["step"])
                now = time.perf_counter()
                if first_step_at is None:
                    first_step_at = now
                last_step_at = now
        elif kind == "error":
            error = str(message.get("message") or "")
    proc.wait()
    log.write_text("".join(lines), encoding="utf-8")

    oom = bool(error) and ("out of gpu memory" in error.lower() or "out of memory" in error.lower())
    # Step 1 pays for the first graph build, so time the interval after it and scale by the gap.
    per_step: float | None = None
    if first_step_at is not None and last_step_at is not None and steps_seen > 1:
        per_step = (last_step_at - first_step_at) / (steps_seen - 1)
    return {
        "peak_vram_gb": vram,
        "seconds_per_step": round(per_step, 2) if per_step else None,
        "seconds_12_steps": round(per_step * _STEPS, 1) if per_step else None,
        "total_seconds": round(time.perf_counter() - started, 1),
        "steps_completed": steps_seen,
        "first_loss": round(losses[0], 4) if losses else None,
        "last_loss": round(losses[-1], 4) if losses else None,
        "status": "oom" if oom else ("ok" if proc.returncode == 0 else "failed"),
        "error": error,
        "log": log.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="dir of NNNN.jpg + NNNN.txt")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--models", type=Path, default=_CORE / "models")
    parser.add_argument("--python", default=str(_CORE / ".venv" / "bin" / "python"))
    parser.add_argument("--gpu", default="", help="label for the results file, e.g. 'L40S (46GB)'")
    parser.add_argument("--only", default="", help="one resolution, to redo a single row")
    parser.add_argument("--steps", type=int, default=_STEPS, help="override for a smoke run")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise SystemExit(f"dataset not found: {args.dataset}")

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    results: dict[str, dict[str, object]] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text()).get("cells", {})

    label = args.gpu
    if not label:
        try:
            label = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001 - the label is cosmetic
            label = "unknown GPU"

    def write() -> None:
        results_path.write_text(
            json.dumps(
                {"gpu": label, "steps": _STEPS, "rank": _RANK, "cells": results}, indent=2
            )
        )

    for resolution in CELLS:
        cell_id = str(resolution)
        if args.only and args.only != cell_id:
            continue
        work = args.out / cell_id
        work.mkdir(parents=True, exist_ok=True)
        manifest = _manifest(work, args.dataset, args.models, resolution, args.steps)
        print(f"--- {cell_id}px, 4-bit base ---", flush=True)
        result = _run_cell(args.python, manifest, args.out / f"{cell_id}.log")
        result.update({"resolution": resolution, "base_quant": "nf4"})
        results[cell_id] = result
        print(json.dumps(result, indent=2), flush=True)
        write()

    write()
    print(f"\n{results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
