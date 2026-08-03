"""VRAM + step-time sweep for FLUX.2 LoRA training: the cells behind the README benchmark table.

    cd core && PYTHONPATH=src .venv/bin/python scripts/flux2_train_matrix.py --dataset <dir>

One cell = one real run of `python -m inline_core.training`, the same entry point the Trainer tab
spawns, so a number here is a number a user would see. Anything else (importing `train` in-process,
or a hand-rolled loop) would measure a different program.

Held fixed at the settings the existing Z-Image and Krea 2 rows used: 12 steps, rank 16, batch 1,
gradient checkpointing on. What varies is resolution and base precision. Peak VRAM is the trainer's
own `torch.cuda.max_memory_allocated` reading off the last progress line; an OOM is recorded as a
cell rather than aborting the sweep, because "does not fit" is a result the table needs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_CORE = _REPO / "core"
_DEFAULT_OUT = _REPO / "outputs" / "flux2-train-matrix"

# (resolution, baseQuant). `none` is the bf16 base; `nf4` is the 4-bit (QLoRA) base.
CELLS: tuple[tuple[int, str], ...] = (
    (512, "none"),
    (512, "nf4"),
    (1024, "none"),
    (1024, "nf4"),
)

_STEPS = 12
_RANK = 16


def _manifest(work: Path, dataset: Path, models: Path, resolution: int, quant: str) -> Path:
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
        "arch": "flux2",
        # FLUX.2 offers one base mode: the undistilled klein base. `raw` is that mode's key.
        "baseMode": "raw",
        "triggerWord": "",
        "hyperparams": {
            "arch": "flux2",
            "baseMode": "raw",
            "baseQuant": quant,
            # Explicit, not `auto`: the sweep is measuring what each precision costs, and auto would
            # silently swap a bf16 cell for NF4 the moment it predicted a bad fit.
            "offload": "off",
            "loraScope": "full",
            "captionDropout": 0.0,
            "flipAugment": False,
            "rank": _RANK,
            "alpha": _RANK,
            "learningRate": 1e-4,
            "batchSize": 1,
            "steps": _STEPS,
            "saveEvery": _STEPS,
            "resolution": resolution,
        },
        "gpuIds": [],
    }
    path = work / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _run_cell(python: str, manifest: Path, log: Path) -> dict[str, object]:
    """Drain the JSON-line protocol, keeping the last VRAM reading and the wall time from the first
    training step onward - loading and latent precache are not what the table reports."""
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
    parser.add_argument("--only", default="", help="substring of a cell id, to redo one row")
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
            name = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            label = name.splitlines()[0]
        except Exception:  # noqa: BLE001 - the label is cosmetic
            label = "unknown GPU"

    def write() -> None:
        results_path.write_text(
            json.dumps(
                {"gpu": label, "steps": _STEPS, "rank": _RANK, "cells": results},
                indent=2,
            )
        )

    for resolution, quant in CELLS:
        cell_id = f"{resolution}-{quant}"
        if args.only and args.only not in cell_id:
            continue
        work = args.out / cell_id
        work.mkdir(parents=True, exist_ok=True)
        manifest = _manifest(work, args.dataset, args.models, resolution, quant)
        print(f"--- {cell_id}: {resolution}px, base {quant} ---", flush=True)
        result = _run_cell(args.python, manifest, args.out / f"{cell_id}.log")
        result.update({"resolution": resolution, "base_quant": quant})
        results[cell_id] = result
        print(json.dumps(result, indent=2), flush=True)
        write()

    write()
    print(f"\n{results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
