"""Core-node generation on the single-process path — ported from the Studio
``electron/main/generation/coreExecutor.ts``, but running the graph through Core's in-process
``RunManager`` instead of over HTTP.

``run_workflow`` serializes a canvas node's upstream closure, submits it, and drains the run's event
stream, translating Core run events into the Studio generation events the SPA listens for
(``events:generationProgress`` / ``NodeDone`` / ``Done`` / ``Error``). Each produced take is copied
into the project's ``takes/`` dir and stored as its Core node's output. Fire-and-forget: the call
returns immediately; progress arrives over ``/events``.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Any

from ..graph.schema import parse_graph
from ..runtime.progress import (
    CancelledEvent,
    ErrorEvent,
    NodeDoneEvent,
    ProgressEvent,
    RunDoneEvent,
)
from . import moodboard as mb
from .graph_build import build_workflow_graph

_EXT = {"image": ".png", "video": ".mp4", "audio": ".mp3"}


def _kind_str(kind: Any) -> str:
    return kind.value if hasattr(kind, "value") else str(kind)


class CoreGeneration:
    """Drives core-node runs and streams their progress as Studio generation events."""

    def __init__(self, store: Any, manager: Any, events: Any) -> None:
        self._store = store
        self._manager = manager
        self._events = events
        self._active: dict[str, str] = {}  # canvas item id -> run id

    def run_workflow(self, item_id: str) -> None:
        graph_dict, target = build_workflow_graph(
            self._store.conn(), self._store.folder(), item_id
        )
        self._progress(item_id, 0.02, "Submitting")
        record, _created = self._manager.submit(parse_graph(graph_dict), target)
        self._active[item_id] = record.state.run_id
        asyncio.create_task(self._drain(item_id, record))

    def cancel(self, item_id: str | None = None) -> None:
        ids = [item_id] if item_id else list(self._active.keys())
        for iid in ids:
            run_id = self._active.pop(iid, None)
            if run_id:
                self._manager.cancel(run_id)

    async def _drain(self, item_id: str, record: Any) -> None:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        record.subscribers.add(queue)
        seen: set[str] = set()
        result = "done"
        try:
            # Snapshot takes produced before we subscribed (fast/cached runs), deduped by id.
            for take in list(record.state.takes):
                if take.id not in seen:
                    seen.add(take.id)
                    self._save_take(item_id, take)
            if record.done:
                self._events.broadcast("events:generationDone", {"targetFrameId": item_id})
                return
            while True:
                event = await queue.get()
                if event is None:  # terminal sentinel
                    break
                if isinstance(event, ProgressEvent):
                    self._progress(item_id, max(0.05, event.fraction), event.status or None)
                elif isinstance(event, NodeDoneEvent):
                    for take in event.takes:
                        if take.id in seen:
                            continue
                        seen.add(take.id)
                        self._save_take(item_id, take)
                        self._events.broadcast(
                            "events:generationNodeDone", {"frameId": item_id, "takeId": take.id}
                        )
                elif isinstance(event, RunDoneEvent):
                    break
                elif isinstance(event, CancelledEvent):
                    result = "cancelled"
                    break
                elif isinstance(event, ErrorEvent):
                    result = "error"
                    self._events.broadcast(
                        "events:generationError",
                        {
                            "targetFrameId": item_id,
                            "frameId": event.node_id,
                            "error": event.message,
                        },
                    )
                    break
            if result == "done":
                self._events.broadcast("events:generationDone", {"targetFrameId": item_id})
        except Exception as error:  # noqa: BLE001
            self._events.broadcast(
                "events:generationError", {"targetFrameId": item_id, "error": str(error)}
            )
        finally:
            record.subscribers.discard(queue)
            self._active.pop(item_id, None)

    def _progress(self, item_id: str, fraction: float, status: str | None) -> None:
        self._events.broadcast(
            "events:generationProgress",
            {"frameId": item_id, "fraction": fraction, "status": status},
        )

    def _save_take(self, item_id: str, take: Any) -> None:
        """Copy a take's bytes into the project's takes/ dir and set its Core node's output."""
        folder: Path = self._store.folder()
        kind = _kind_str(take.kind)
        src = Path(take.uri)
        ext = src.suffix or _EXT.get(kind, ".png")
        rel = f"takes/{uuid.uuid4()}{ext}"
        (folder / "takes").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, folder / rel)
        # take.node_id is the canvas item that produced it (node ids == item ids).
        mb.set_core_node_output(
            self._store.conn(), take.node_id, {"takeId": take.id, "filePath": rel, "kind": kind}
        )
