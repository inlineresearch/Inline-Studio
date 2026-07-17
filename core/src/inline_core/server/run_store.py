"""Durable run storage (SQLite, stdlib). A runId and its final state survive a process restart, so
GET /v1/runs/{id} keeps working after Core is bounced. Runs left mid-flight by a crash are marked
interrupted on the next start. Live progress ticks are not persisted (they are lost on a crash
anyway); the record captures structure, terminal status, and takes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from ..media import MediaKind
from ..runtime.run import NodeRuntimeState, NodeState, RunError, RunState, RunStatus
from ..takes import Take


class RunStore(ABC):
    @abstractmethod
    def interrupt_stale(self) -> None: ...

    @abstractmethod
    def create(self, state: RunState, client_run_id: str | None) -> None: ...

    @abstractmethod
    def update(self, state: RunState) -> None: ...

    @abstractmethod
    def load(self, run_id: str) -> RunState | None: ...

    @abstractmethod
    def find_take(self, take_id: str) -> Take | None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, target TEXT, status TEXT, fraction REAL,
  error_message TEXT, error_node TEXT, client_run_id TEXT, created_at INTEGER
);
CREATE TABLE IF NOT EXISTS run_nodes (
  run_id TEXT, node_id TEXT, state TEXT, phase TEXT, fraction REAL,
  step INTEGER, step_count INTEGER, status TEXT, PRIMARY KEY (run_id, node_id)
);
CREATE TABLE IF NOT EXISTS run_takes (
  take_id TEXT PRIMARY KEY, run_id TEXT, node_id TEXT, kind TEXT, uri TEXT,
  hash TEXT, params TEXT, created_at INTEGER
);
"""


class SqliteRunStore(RunStore):
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def interrupt_stale(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET status=?, error_message=? WHERE status IN ('queued','running')",
                (RunStatus.ERROR.value, "interrupted by a restart"),
            )

    def create(self, state: RunState, client_run_id: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (state.run_id, state.target, state.status.value, state.fraction, None, None,
                 client_run_id, 0),
            )
            self._write_nodes(state)

    def update(self, state: RunState) -> None:
        error_message = state.error.message if state.error is not None else None
        error_node = state.error.node_id if state.error is not None else None
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET status=?, fraction=?, error_message=?, error_node=? "
                "WHERE run_id=?",
                (state.status.value, state.fraction, error_message, error_node, state.run_id),
            )
            self._write_nodes(state)
            self._conn.execute("DELETE FROM run_takes WHERE run_id=?", (state.run_id,))
            self._conn.executemany(
                "INSERT OR REPLACE INTO run_takes VALUES (?,?,?,?,?,?,?,?)",
                [
                    (t.id, t.run_id, t.node_id, t.kind.value, t.uri, t.hash,
                     json.dumps(t.params), t.created_at)
                    for t in state.takes
                ],
            )

    def _write_nodes(self, state: RunState) -> None:
        self._conn.execute("DELETE FROM run_nodes WHERE run_id=?", (state.run_id,))
        self._conn.executemany(
            "INSERT OR REPLACE INTO run_nodes VALUES (?,?,?,?,?,?,?,?)",
            [
                (state.run_id, node_id, n.state.value, n.phase, n.fraction, n.step, n.step_count,
                 n.status)
                for node_id, n in state.nodes.items()
            ],
        )

    def load(self, run_id: str) -> RunState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT target, status, fraction, error_message, error_node FROM runs "
                "WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            node_rows = self._conn.execute(
                "SELECT node_id, state, phase, fraction, step, step_count, status FROM run_nodes "
                "WHERE run_id=?",
                (run_id,),
            ).fetchall()
            take_rows = self._conn.execute(
                "SELECT take_id, node_id, kind, uri, hash, params, created_at FROM run_takes "
                "WHERE run_id=?",
                (run_id,),
            ).fetchall()

        target, status, fraction, error_message, error_node = row
        state = RunState(run_id=run_id, target=target, status=RunStatus(status), fraction=fraction)
        for node_id, node_state, phase, node_fraction, step, step_count, node_status in node_rows:
            state.nodes[node_id] = NodeRuntimeState(
                state=NodeState(node_state), phase=phase, fraction=node_fraction,
                step=step, step_count=step_count, status=node_status,
            )
        for take_id, node_id, kind, uri, take_hash, params, created_at in take_rows:
            state.takes.append(
                Take(id=take_id, run_id=run_id, node_id=node_id, kind=MediaKind(kind),
                     uri=uri, hash=take_hash, params=json.loads(params), created_at=created_at)
            )
        if error_message is not None:
            state.error = RunError(message=error_message, node_id=error_node)
        return state

    def find_take(self, take_id: str) -> Take | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, node_id, kind, uri, hash, params, created_at FROM run_takes "
                "WHERE take_id=?",
                (take_id,),
            ).fetchone()
        if row is None:
            return None
        run_id, node_id, kind, uri, take_hash, params, created_at = row
        return Take(id=take_id, run_id=run_id, node_id=node_id, kind=MediaKind(kind),
                    uri=uri, hash=take_hash, params=json.loads(params), created_at=created_at)
