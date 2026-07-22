"""The JSON-line progress protocol between the trainer subprocess and the Studio orchestrator.

Every message is one JSON object on its own stdout line. Keep these shapes in sync with the
orchestrator's parser (``studio/training.py`` ``_drain`` / ``_on_progress``). Torch-free on purpose
so the entry point can report a clean error even when the training stack is missing.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def progress(
    step: int, total: int, *, loss: float | None = None, status: str | None = None
) -> None:
    message: dict[str, Any] = {
        "type": "progress",
        "step": step,
        "total": total,
        "fraction": (step / total) if total else 0.0,
    }
    if loss is not None:
        message["loss"] = loss
    if status is not None:
        message["status"] = status
    emit(message)


def sample(step: int, path: str) -> None:
    emit({"type": "sample", "step": step, "path": path})


def checkpoint(path: str) -> None:
    emit({"type": "checkpoint", "path": path})


def done(output: str) -> None:
    emit({"type": "done", "output": output})


def error(message: str) -> None:
    emit({"type": "error", "message": message})
