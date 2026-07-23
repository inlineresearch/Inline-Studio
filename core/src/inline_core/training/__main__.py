"""Trainer entry point: ``python -m inline_core.training <manifest.json>``.

Reads the run manifest the orchestrator wrote, runs the LoRA training loop, and reports progress as
JSON lines on stdout (``protocol.py``). The heavy ``trainer`` import is deferred to ``main`` so a
bad invocation - or an install without the ``training`` extra - reports a clean error instead of an
``ImportError`` traceback. A SIGTERM (the orchestrator's cancel) asks the loop to flush a final
checkpoint and stop; the run is then resumable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import protocol


def _message_for(error: Exception, manifest: dict[str, Any]) -> str:
    """Turn a CUDA OOM into something the user can act on. Training resolution drives peak VRAM far
    more than rank does, so point at that first - a 16GB card trains Z-Image at 512 but not 768."""
    if type(error).__name__ != "OutOfMemoryError":
        return str(error)
    resolution = int(manifest.get("hyperparams", {}).get("resolution", 0) or 0)
    lower = " Try 512." if resolution > 512 else ""
    at = f" at {resolution}px" if resolution else ""
    return (
        f"Ran out of GPU memory training{at}. Lower the training resolution in the node's Adjust "
        f"panel (it drives peak VRAM far more than rank).{lower}"
    )


def main(argv: list[str]) -> int:
    if not argv:
        protocol.error("No manifest path given.")
        return 2
    try:
        manifest = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        protocol.error(f"Could not read manifest: {exc}")
        return 2

    try:
        from .trainer import train  # heavy deps (torch/diffusers/peft) load here, not at import
    except Exception as exc:  # noqa: BLE001 - report a missing training stack cleanly
        protocol.error(f"Training stack unavailable: {exc}. Install the 'training' extra.")
        return 1

    try:
        output = train(manifest)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - surface any training failure as one error line
        protocol.error(_message_for(exc, manifest))
        return 1

    if output is None:
        # A cooperative stop (cancel/SIGTERM): a checkpoint was saved, no final LoRA. Non-`done`
        # exit tells the orchestrator this run is cancelled/resumable, not complete.
        return 0
    protocol.done(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
