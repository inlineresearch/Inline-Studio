"""Orchestrate LoRA training: a Studio-side long-running job that spawns the trainer as a subprocess
and streams progress over ``/events`` - mirroring the ffmpeg timeline (``timeline/render.py``).

Training never runs inside Core's graph executor (``core/CLAUDE.md``: the executor never runs the
denoise loop). ``start`` writes the dataset to a working dir + a manifest, launches
``python -m inline_core.training <manifest>`` (or ``accelerate launch --multi_gpu`` for 2+ GPUs),
and returns immediately; the subprocess emits JSON-line progress that this class parses, mirrors
into the durable ``training_runs`` row, and broadcasts. The produced ``.safetensors`` is written
into ``models_dir()/loras/`` so it shows in the LoRA loader node's dropdown - no loader changes.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from ..config import models_dir
from . import training_store as ts

_CAPTION_EXT = ".txt"


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "lora"


class Training:
    """Owns the LoRA training lifecycle: dataset CRUD, auto-caption, and one run at a time."""

    def __init__(self, store: Any, events: Any, on_output: Any = None) -> None:
        self._store = store
        self._events = events
        # Rescans the model catalog when a run's LoRA lands in models/loras/, so it appears in the
        # loader node's dropdown + bumps the registry version - the same hook model downloads use.
        self._on_output = on_output
        # One run holds the GPU at a time; the process is kept so cancel can SIGTERM it.
        self._active: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    def _conn(self) -> Any:
        return self._store.conn()

    # --- dataset CRUD (delegates to training_store) ---------------------------------------------

    def list_datasets(self) -> list[dict[str, Any]]:
        return ts.list_datasets(self._conn())

    def create_dataset(self, inp: dict[str, Any]) -> dict[str, Any]:
        return ts.create_dataset(self._conn(), inp["name"], inp.get("triggerWord") or "")

    def list_items(self, dataset_id: str) -> list[dict[str, Any]]:
        return ts.list_items(self._conn(), dataset_id)

    def add_items(self, dataset_id: str, asset_ids: list[str]) -> list[dict[str, Any]]:
        return ts.add_items(self._conn(), dataset_id, asset_ids)

    def remove_item(self, item_id: str) -> None:
        ts.remove_item(self._conn(), item_id)

    def set_caption(self, item_id: str, caption: str) -> dict[str, Any]:
        return ts.set_caption(self._conn(), item_id, caption)

    def list_runs(self) -> list[dict[str, Any]]:
        return ts.list_runs(self._conn())

    def status(self, run_id: str) -> dict[str, Any]:
        return ts.get_run(self._conn(), run_id)

    # --- auto-caption ---------------------------------------------------------------------------

    async def auto_caption(self, dataset_id: str, overwrite: bool) -> list[dict[str, Any]]:
        """Caption items with the local VLM (a subprocess, so torch never imports server-side)."""
        conn, folder = self._conn(), self._store.folder()
        items = ts.list_items(conn, dataset_id)
        targets = [
            {"id": it["id"], "path": str(folder / self._asset_path(it["assetId"]))}
            for it in items
            if overwrite or not it["caption"].strip()
        ]
        if not targets:
            return items
        manifest = {"items": targets}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "inline_core.training.caption",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate(json.dumps(manifest).encode())
        for line in out.decode(errors="replace").splitlines():
            msg = _parse_json_line(line)
            if msg and msg.get("id") and isinstance(msg.get("caption"), str):
                ts.set_caption(conn, msg["id"], msg["caption"])
        return ts.list_items(conn, dataset_id)

    def _relativize(self, path: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(self._store.folder().resolve()))
        except (ValueError, RuntimeError):
            return path

    def _asset_path(self, asset_id: str) -> str:
        from . import assets as ax

        file = ax.asset_file(self._conn(), asset_id)
        if file is None:
            raise ValueError("Dataset references a missing asset.")
        return file["filePath"]

    # --- training run ---------------------------------------------------------------------------

    def start(self, dataset_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("A training run is already in progress; wait for it to finish.")
        dataset = ts.get_dataset(self._conn(), dataset_id)
        run = ts.create_run(self._conn(), dataset_id, dataset["name"], hyperparams)
        asyncio.create_task(self._run(run["id"], resume=False))
        return run

    def resume(self, run_id: str) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("A training run is already in progress; wait for it to finish.")
        run = ts.get_run(self._conn(), run_id)
        if run["status"] not in ("interrupted", "failed"):
            raise RuntimeError("Only an interrupted run can be resumed.")
        asyncio.create_task(self._run(run_id, resume=True))
        return ts.update_run(self._conn(), run_id, {"status": "queued", "error": None})

    def cancel(self, run_id: str) -> None:
        proc = self._active.get(run_id)
        if proc is not None:
            self._cancelled.add(run_id)
            proc.terminate()  # SIGTERM -> the trainer flushes a final checkpoint before exit

    async def _run(self, run_id: str, *, resume: bool) -> None:
        conn = self._conn()
        try:
            manifest_path, output_rel = self._prepare(run_id, resume=resume)
        except Exception as error:  # noqa: BLE001 - surface prep failures as a run error
            ts.update_run(conn, run_id, {"status": "failed", "error": str(error)})
            self._events.broadcast("events:trainingError", {"runId": run_id, "error": str(error)})
            return

        ts.update_run(conn, run_id, {"status": "training", "progressStatus": "starting"})
        proc = await self._spawn(run_id, manifest_path)
        self._active[run_id] = proc
        saw_done = False
        try:
            saw_done = await self._drain(run_id, proc, output_rel)
        finally:
            await proc.wait()
            self._active.pop(run_id, None)
        self._finish(run_id, proc.returncode or 0, saw_done, output_rel)

    async def _drain(self, run_id: str, proc: asyncio.subprocess.Process, output_rel: str) -> bool:
        """Read the trainer's JSON-line stdout, mirroring progress into the row + events."""
        saw_done = False
        assert proc.stdout is not None
        async for raw in proc.stdout:
            msg = _parse_json_line(raw.decode(errors="replace"))
            if msg is None:
                continue
            kind = msg.get("type")
            if kind == "progress":
                self._on_progress(run_id, msg)
            elif kind == "sample" and msg.get("path"):
                # /media serves project-relative paths, so relativize the trainer's absolute one.
                self._events.broadcast(
                    "events:trainingSample",
                    {
                        "runId": run_id,
                        "step": int(msg.get("step", 0)),
                        "path": self._relativize(msg["path"]),
                    },
                )
            elif kind == "checkpoint" and msg.get("path"):
                ts.update_run(self._conn(), run_id, {"checkpointPath": msg["path"]})
            elif kind == "done":
                saw_done = True
        return saw_done

    def _on_progress(self, run_id: str, msg: dict[str, Any]) -> None:
        step = int(msg.get("step", 0))
        total = int(msg.get("total", 0)) or None
        fraction = float(msg.get("fraction", 0.0))
        status = msg.get("status")
        patch: dict[str, Any] = {"progressFraction": fraction, "step": step}
        if total:
            patch["totalSteps"] = total
        if status:
            patch["progressStatus"] = status
        ts.update_run(self._conn(), run_id, patch)
        event: dict[str, Any] = {
            "runId": run_id, "fraction": fraction, "step": step, "totalSteps": total or 0,
        }
        if isinstance(msg.get("loss"), int | float):
            event["loss"] = float(msg["loss"])
        if status:
            event["status"] = status
        self._events.broadcast("events:trainingProgress", event)

    def _finish(self, run_id: str, code: int, saw_done: bool, output_rel: str) -> None:
        conn = self._conn()
        if run_id in self._cancelled:
            self._cancelled.discard(run_id)
            # SIGTERM flushes a final checkpoint (trainer.py: "a resumable cancel"), so a cancel
            # that got far enough to checkpoint is offered for Resume (the status the UI shows a
            # Resume on), not a dead-end "cancelled". A cancel before any checkpoint stays terminal.
            run = ts.get_run(conn, run_id)
            resumable = bool(run.get("checkpointPath")) or run["step"] > 0
            if resumable:
                ts.update_run(
                    conn, run_id,
                    {"status": "interrupted", "progressStatus": "cancelled",
                     "error": "Training was cancelled; you can resume it."},
                )
                self._events.broadcast(
                    "events:trainingError", {"runId": run_id, "error": "Cancelled; resumable."}
                )
            else:
                ts.update_run(conn, run_id, {"status": "cancelled", "progressStatus": "cancelled"})
                self._events.broadcast(
                    "events:trainingError", {"runId": run_id, "error": "Cancelled."}
                )
            return
        if saw_done and code == 0 and (models_dir() / output_rel).is_file():
            ts.update_run(
                conn, run_id,
                {"status": "done", "progressFraction": 1.0, "outputLoraPath": output_rel},
            )
            if self._on_output is not None:
                self._on_output()  # rescan so the new LoRA shows in the loader dropdown
            self._events.broadcast(
                "events:trainingDone", {"runId": run_id, "outputLoraPath": output_rel}
            )
            return
        # Non-zero exit with a checkpoint is resumable; otherwise it failed outright.
        run = ts.get_run(conn, run_id)
        resumable = bool(run.get("checkpointPath")) or run["step"] > 0
        status = "interrupted" if resumable else "failed"
        error = "Training was interrupted; you can resume it." if resumable else "Training failed."
        ts.update_run(conn, run_id, {"status": status, "error": error})
        self._events.broadcast("events:trainingError", {"runId": run_id, "error": error})

    # --- subprocess plumbing --------------------------------------------------------------------

    def _prepare(self, run_id: str, *, resume: bool) -> tuple[Path, str]:
        """Export the dataset + write the run manifest. Returns (manifest path, loras-rel path)."""
        conn, folder = self._conn(), self._store.folder()
        run = ts.get_run(conn, run_id)
        dataset = ts.get_dataset(conn, run["datasetId"])
        items = ts.list_items(conn, run["datasetId"])
        if not items:
            raise ValueError("Add at least one image to the dataset before training.")

        work = folder / "training_runs" / run_id
        dataset_dir = work / "dataset"
        checkpoint_dir = work / "checkpoints"
        for d in (dataset_dir, checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)

        trigger = dataset["triggerWord"].strip()
        for i, item in enumerate(items):
            src = folder / self._asset_path(item["assetId"])
            dest = dataset_dir / f"{i:04d}{src.suffix.lower()}"
            shutil.copyfile(src, dest)
            caption = ", ".join(p for p in (trigger, item["caption"].strip()) if p)
            dest.with_suffix(_CAPTION_EXT).write_text(caption, encoding="utf-8")

        loras = models_dir() / "loras"
        loras.mkdir(parents=True, exist_ok=True)
        output_rel = f"loras/{_safe(run['name'])}-{run_id[:8]}.safetensors"
        resume_from = str(checkpoint_dir) if resume else None
        manifest = {
            "runId": run_id,
            "workingDir": str(work),
            "datasetDir": str(dataset_dir),
            "checkpointDir": str(checkpoint_dir),
            "outputPath": str(models_dir() / output_rel),
            "resumeFrom": resume_from,
            "modelsDir": str(models_dir()),
            "baseMode": run["hyperparams"]["baseMode"],
            "triggerWord": trigger,
            "hyperparams": run["hyperparams"],
            "gpuIds": run["hyperparams"].get("gpuIds") or [],
        }
        manifest_path = work / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path, output_rel

    async def _spawn(
        self, run_id: str, manifest_path: Path
    ) -> asyncio.subprocess.Process:
        import os

        run = ts.get_run(self._conn(), run_id)
        gpu_ids = [int(g) for g in (run["hyperparams"].get("gpuIds") or [])]
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        base = [sys.executable]
        if len(gpu_ids) > 1:
            # DDP: only the small LoRA gradients all-reduce, so scaling is near-linear.
            base += ["-m", "accelerate.commands.launch", "--multi_gpu",
                     "--num_processes", str(len(gpu_ids))]
        cmd = [*base, "-m", "inline_core.training", str(manifest_path)]
        return await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
        )


def _parse_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
