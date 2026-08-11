"""One view of everything running: Core generations, fal generations, and training runs.

Studio flattened every run to ``{frameId, fraction, status}``, which left no way to tell queued from
running, no run id to cancel by, and no record of which project a run belongs to. This module keeps
that state in one place for the process lifetime, so the UI can show activity across projects and
both tabs while Core keeps working through a project switch.

Engine runs arrive through ``RunManager``'s observer seam rather than being reported by the Studio
call that started them, so a run submitted straight to ``POST /v1/runs`` shows up here too.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..runtime.run import RunStatus
from .store import ProjectRef, StudioStore

RunKind = Literal["generation", "training"]
RunEngine = Literal["core", "fal"]
RunOrigin = Literal["studio", "api"]
ActivityStatus = Literal["queued", "running", "done", "error", "cancelled", "interrupted"]

# Terminal states, the ones that move a run out of the live list and into project history.
_TERMINAL: frozenset[str] = frozenset({"done", "error", "cancelled", "interrupted"})

# Live-list broadcasts per second. Progress already coalesces upstream in RunManager; this only
# stops several concurrent runs from multiplying that rate.
_MIN_BROADCAST_INTERVAL = 0.25

# Finished runs kept in memory. History lives in the project db; this is just what a client sees
# without asking for it.
_RECENT_LIMIT = 50


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ActivityRun:
    run_id: str
    kind: RunKind
    engine: RunEngine
    origin: RunOrigin
    status: ActivityStatus
    title: str
    queued_at: int
    fraction: float | None = None
    status_label: str | None = None
    queue_position: int | None = None
    started_at: int | None = None
    ended_at: int | None = None
    error: str | None = None
    take_id: str | None = None
    # Studio-origin only. An API run has no project and no canvas node behind it.
    project_id: str | None = None
    project_name: str | None = None
    project_path: str | None = None
    item_id: str | None = None
    surface: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "kind": self.kind,
            "engine": self.engine,
            "origin": self.origin,
            "status": self.status,
            "title": self.title,
            "fraction": self.fraction,
            "statusLabel": self.status_label,
            "queuePosition": self.queue_position,
            "queuedAt": self.queued_at,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "error": self.error,
            "takeId": self.take_id,
            "projectId": self.project_id,
            "projectName": self.project_name,
            "projectPath": self.project_path,
            "itemId": self.item_id,
            "surface": self.surface,
        }


def _row_to_json(row: Any) -> dict[str, Any]:
    return {
        "runId": row["id"],
        "kind": "generation",
        "engine": row["engine"],
        "origin": "studio",
        "status": row["status"],
        "title": row["title"],
        "fraction": 1.0 if row["status"] == "done" else None,
        "statusLabel": None,
        "queuePosition": None,
        "queuedAt": row["queued_at"],
        "startedAt": row["started_at"],
        "endedAt": row["ended_at"],
        "error": row["error"],
        "takeId": row["take_id"],
        "projectId": row["project_id"],
        "projectName": None,
        "projectPath": None,
        "itemId": row["item_id"],
        "surface": row["surface"],
    }


class ActivityRegistry:
    """Live runs for the whole process, plus per-project history in each project's own db."""

    def __init__(self, store: StudioStore, events: Any) -> None:
        self._store = store
        self._events = events
        self._runs: dict[str, ActivityRun] = {}
        self._refs: dict[str, ProjectRef] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_broadcast = 0.0
        self._cancellers: dict[str, Callable[[str], None]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_canceller(self, route: str, cancel: Callable[[str], None]) -> None:
        """Register how to cancel one family of run: `core`, `fal` or `training`."""
        self._cancellers[route] = cancel

    # --- reads ----------------------------------------------------------------------------------

    def live(self) -> list[ActivityRun]:
        """Queued and running, oldest first. What the indicator counts."""
        with self._lock:
            return [
                self._runs[rid]
                for rid in self._order
                if rid in self._runs and self._runs[rid].status not in _TERMINAL
            ]

    def recent(self) -> list[ActivityRun]:
        with self._lock:
            done = [
                self._runs[rid]
                for rid in self._order
                if rid in self._runs and self._runs[rid].status in _TERMINAL
            ]
        return done[-_RECENT_LIMIT:][::-1]

    def snapshot(self) -> list[dict[str, Any]]:
        return [r.to_json() for r in self.live()]

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Finished runs for the open project. Returns nothing when no project is open."""
        ref = self._store.project_ref()
        if ref is None:
            return []
        with self._store.bind(ref) as conn:
            rows = conn.execute(
                "SELECT * FROM generation_runs WHERE project_id = ? "
                "ORDER BY queued_at DESC LIMIT ?",
                (ref.id, max(1, min(limit, 200))),
            ).fetchall()
        return [_row_to_json(r) for r in rows]

    def clear_history(self) -> None:
        ref = self._store.project_ref()
        if ref is None:
            return
        with self._store.bind(ref) as conn:
            conn.execute("DELETE FROM generation_runs WHERE project_id = ?", (ref.id,))

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            return False
        # Core and fal runs are both "generation" but cancel through different machinery.
        route = "training" if run.kind == "training" else run.engine
        cancel = self._cancellers.get(route)
        if cancel is None:
            return False
        cancel(run_id)
        return True

    # --- writes ---------------------------------------------------------------------------------

    def track(self, run: ActivityRun, ref: ProjectRef | None = None) -> None:
        with self._lock:
            if run.run_id not in self._runs:
                self._order.append(run.run_id)
            self._runs[run.run_id] = run
            if ref is not None:
                self._refs[run.run_id] = ref
        self._broadcast(force=True)

    def update(self, run_id: str, **fields: Any) -> None:
        transition = "status" in fields
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            for key, value in fields.items():
                setattr(run, key, value)
        self._broadcast(force=transition)

    def finish(
        self,
        run_id: str,
        status: ActivityStatus,
        *,
        error: str | None = None,
        take_id: str | None = None,
    ) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = status
            run.ended_at = _now_ms()
            run.queue_position = None
            run.fraction = 1.0 if status == "done" else run.fraction
            if error is not None:
                run.error = error
            if take_id is not None:
                run.take_id = take_id
            ref = self._refs.pop(run_id, None)
            snapshot = run
        if ref is not None:
            self._write_history(snapshot, ref)
        self._prune()
        self._broadcast(force=True)

    def _write_history(self, run: ActivityRun, ref: ProjectRef) -> None:
        # Writes against the project the run was submitted for, which may no longer be the open one.
        try:
            with self._store.bind(ref) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO generation_runs (id, project_id, item_id, surface, "
                    "engine, title, status, error, take_id, queued_at, started_at, ended_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run.run_id,
                        ref.id,
                        run.item_id or "",
                        run.surface or "studio",
                        run.engine,
                        run.title,
                        run.status,
                        run.error,
                        run.take_id,
                        run.queued_at,
                        run.started_at,
                        run.ended_at,
                    ),
                )
        except Exception:  # noqa: BLE001 - history is not worth failing a finished run over
            pass

    def reconcile(self) -> None:
        """Flip rows left mid-run by a crash to `interrupted`, so history has no stuck 'running'."""
        ref = self._store.project_ref()
        if ref is None:
            return
        with self._lock:
            alive = {rid for rid, r in self._runs.items() if r.status not in _TERMINAL}
        with self._store.bind(ref) as conn:
            rows = conn.execute(
                "SELECT id FROM generation_runs WHERE project_id = ? AND status IN "
                "('queued','running')",
                (ref.id,),
            ).fetchall()
            stale = [r["id"] for r in rows if r["id"] not in alive]
            for run_id in stale:
                conn.execute(
                    "UPDATE generation_runs SET status = 'interrupted', ended_at = ? WHERE id = ?",
                    (_now_ms(), run_id),
                )

    def _prune(self) -> None:
        with self._lock:
            finished = [
                rid
                for rid in self._order
                if rid in self._runs and self._runs[rid].status in _TERMINAL
            ]
            for run_id in finished[:-_RECENT_LIMIT]:
                self._runs.pop(run_id, None)
                self._order.remove(run_id)

    # --- engine seam ----------------------------------------------------------------------------

    def observe(self, manager: Any) -> None:
        manager.add_observer(lambda phase, record: self._on_engine(manager, phase, record))

    def _on_engine(self, manager: Any, phase: str, record: Any) -> None:
        run_id = record.state.run_id
        if phase == "queued":
            meta = record.meta
            project_id = meta.get("projectId")
            self.track(
                ActivityRun(
                    run_id=run_id,
                    kind="generation",
                    engine="core",
                    origin="studio" if project_id else "api",
                    status="queued",
                    title=str(meta.get("title") or record.state.target),
                    queued_at=_now_ms(),
                    queue_position=manager.queue_position(run_id),
                    project_id=str(project_id) if project_id else None,
                    project_name=_opt(meta.get("projectName")),
                    project_path=_opt(meta.get("projectPath")),
                    item_id=_opt(meta.get("itemId")),
                    surface=_opt(meta.get("surface")),
                ),
                ref=_ref_from_meta(record.meta),
            )
            return
        if phase == "started":
            self.update(run_id, status="running", started_at=_now_ms(), queue_position=None)
            return
        if phase == "progress":
            self.update(
                run_id,
                fraction=record.state.fraction,
                status_label=_running_label(record.state),
            )
            return
        status = record.state.status
        if status is RunStatus.CANCELLED:
            self.finish(run_id, "cancelled")
        elif status is RunStatus.ERROR:
            message = record.state.error.message if record.state.error else "Run failed."
            self.finish(run_id, "error", error=message)
        else:
            take_id = record.state.takes[-1].id if record.state.takes else None
            self.finish(run_id, "done", take_id=take_id)

    # --- broadcast ------------------------------------------------------------------------------

    def _broadcast(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_broadcast < _MIN_BROADCAST_INTERVAL:
            return
        self._last_broadcast = now
        payload = {"runs": self.snapshot()}
        loop = self._loop
        # Observers run on the manager's worker thread, and the broadcaster feeds asyncio queues,
        # so anything off the loop has to be handed back to it.
        if loop is None or _on_loop(loop):
            self._events.broadcast("events:activityChanged", payload)
            return
        loop.call_soon_threadsafe(self._events.broadcast, "events:activityChanged", payload)


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _opt(value: Any) -> str | None:
    return str(value) if value else None


def _ref_from_meta(meta: dict[str, Any]) -> ProjectRef | None:
    pid, name, path = meta.get("projectId"), meta.get("projectName"), meta.get("projectPath")
    if not pid or not path:
        return None
    return ProjectRef(id=str(pid), name=str(name or ""), folder=Path(str(path)))


def _running_label(state: Any) -> str | None:
    for node in state.nodes.values():
        if node.status:
            return str(node.status)
    return None
