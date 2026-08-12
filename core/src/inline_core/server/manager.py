"""The run manager: validate, queue, execute on a worker thread, fan out events to subscribers.

The durable RunState is authoritative (GET /v1/runs). The websocket stream is a fan-out on top; the
state is updated on the worker thread (via the executor's StateTrackingEmitter) before each publish.
A RunStore persists runs so they survive a restart; progress events are coalesced to the stream.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from uuid import uuid4

from ..device.memory import MemoryPolicy
from ..device.policy import DevicePolicy
from ..errors import InlineCoreError
from ..graph.cache import NodeCache
from ..graph.executor import Executor
from ..graph.registry import Registry
from ..graph.schema import Graph
from ..graph.topo import topo_sort, upstream_closure
from ..graph.validate import validate
from ..runtime.context import CancelToken, ExecutionContext
from ..runtime.progress import (
    CancelledEvent,
    NodeDoneEvent,
    ProgressEmitter,
    ProgressEvent,
    RunEvent,
    RunStartedEvent,
)
from ..runtime.run import NodeRuntimeState, RunState, RunStatus, apply_event
from ..runtime.store import TakeStore
from ..takes import Take
from .run_store import RunStore

# ~10 progress events per second per run to the stream (contract section 6). The snapshot stays
# authoritative, so dropping intermediate ticks only affects stream chattiness.
_MIN_PROGRESS_INTERVAL = 0.1

# What an observer is told about. Every run reaches the worker through this manager whoever
# submitted it, so this is the one seam that sees API runs and Studio runs alike.
RunPhase = Literal["queued", "started", "progress", "finished"]
RunObserver = Callable[[RunPhase, "RunRecord"], None]


class RunConflict(InlineCoreError):
    """A clientRunId was reused with a different graph."""


class RunRecord:
    def __init__(
        self, state: RunState, cancel: CancelToken, meta: dict[str, object] | None = None
    ) -> None:
        self.state = state
        self.cancel = cancel
        # Opaque to the engine: whatever the caller wants carried alongside the run.
        self.meta: dict[str, object] = dict(meta or {})
        self.subscribers: set[asyncio.Queue[RunEvent | None]] = set()
        self.done = False
        self.last_progress = 0.0
        self.queued_at = time.time()


class _BroadcastEmitter(ProgressEmitter):
    def __init__(self, manager: RunManager, record: RunRecord) -> None:
        self._manager = manager
        self._record = record

    def emit(self, event: RunEvent) -> None:
        self._manager.publish(self._record, event)


class RunManager:
    def __init__(
        self,
        registry: Registry,
        cache: NodeCache,
        policy: DevicePolicy | None = None,
        workers: int = 1,
        store: RunStore | None = None,
        takes: TakeStore | None = None,
    ) -> None:
        self._registry = registry
        self._cache = cache
        self._policy = policy or MemoryPolicy()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self._runs: dict[str, RunRecord] = {}
        self._by_client: dict[str, str] = {}
        self._graph_hash: dict[str, str] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._store = store
        self._takes = takes
        self._observers: list[RunObserver] = []
        self._order: list[str] = []
        if store is not None:
            store.interrupt_stale()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add_observer(self, observer: RunObserver) -> None:
        self._observers.append(observer)

    def _notify(self, phase: RunPhase, record: RunRecord) -> None:
        # An observer must never take the manager down with it, so failures are swallowed here.
        for observer in list(self._observers):
            try:
                observer(phase, record)
            except Exception:  # noqa: BLE001
                pass

    def list_runs(self) -> list[RunRecord]:
        """Live records in submission order, so a caller can render the queue."""
        with self._lock:
            return [self._runs[rid] for rid in self._order if rid in self._runs]

    def queue_position(self, run_id: str) -> int | None:
        """0-based place in the pending line, or None once the run has left the queue."""
        with self._lock:
            pending = [
                rid
                for rid in self._order
                if rid in self._runs and self._runs[rid].state.status is RunStatus.QUEUED
            ]
        return pending.index(run_id) if run_id in pending else None

    def shutdown(self) -> None:
        for record in list(self._runs.values()):
            record.cancel.cancel()
        self._pool.shutdown(wait=False)

    def submit(
        self,
        graph: Graph,
        target: str,
        client_run_id: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> tuple[RunRecord, bool]:
        validate(graph, target, self._registry)
        graph_hash = _graph_hash(graph, target)
        if client_run_id is not None and client_run_id in self._by_client:
            existing = self._runs[self._by_client[client_run_id]]
            if self._graph_hash.get(existing.state.run_id) == graph_hash:
                return existing, False
            raise RunConflict(f"clientRunId {client_run_id!r} was reused with a different graph.")
        run_id = f"run_{uuid4().hex[:12]}"
        state = RunState(run_id=run_id, target=target)
        for node_id in topo_sort(
            list(upstream_closure(target, graph.input_sources)), graph.input_sources
        ):
            state.nodes[node_id] = NodeRuntimeState()
        record = RunRecord(state, CancelToken(), meta)
        with self._lock:
            self._runs[run_id] = record
            self._order.append(run_id)
            if client_run_id is not None:
                self._by_client[client_run_id] = run_id
                self._graph_hash[run_id] = graph_hash
        if self._store is not None:
            self._store.create(state, client_run_id)
        self._notify("queued", record)
        self._pool.submit(self._execute, graph, target, record)
        return record, True

    def get(self, run_id: str) -> RunRecord | None:
        record = self._runs.get(run_id)
        if record is not None:
            return record
        if self._store is not None:
            state = self._store.load(run_id)
            if state is not None:
                return _historical(state)
        return None

    def cancel(self, run_id: str) -> bool:
        record = self._runs.get(run_id)
        if record is None:
            return False
        record.cancel.cancel()
        return True

    def find_take(self, take_id: str) -> Take | None:
        for record in self._runs.values():
            for take in record.state.takes:
                if take.id == take_id:
                    return take
        if self._store is not None:
            return self._store.find_take(take_id)
        return None

    def publish(self, record: RunRecord, event: RunEvent | None) -> None:
        if not self._should_stream(record, event, time.monotonic()):
            return
        # Observers reuse the stream's coalescing rather than adding a second throttle.
        if isinstance(event, ProgressEvent | NodeDoneEvent):
            self._notify("progress", record)
        loop = self._loop
        if loop is None:
            return
        for queue in list(record.subscribers):
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def _should_stream(self, record: RunRecord, event: RunEvent | None, now: float) -> bool:
        """Coalesce progress ticks; node_done, terminal events, and the finish sentinel pass."""
        if isinstance(event, ProgressEvent):
            if now - record.last_progress < _MIN_PROGRESS_INTERVAL:
                return False
            record.last_progress = now
        return True

    def _execute(self, graph: Graph, target: str, record: RunRecord) -> None:
        if record.cancel.cancelled:
            # Cancelled while still queued: skip the context entirely, but still land a terminal
            # state so subscribers are not left waiting on a run that will never start.
            self._emit(record, CancelledEvent(run_id=record.state.run_id))
            self._settle(record)
            return
        self._emit(record, RunStartedEvent(run_id=record.state.run_id))
        self._notify("started", record)
        ctx = ExecutionContext(
            run_id=record.state.run_id,
            policy=self._policy,
            emitter=_BroadcastEmitter(self, record),
            cancel=record.cancel,
            takes=self._takes,
        )
        try:
            Executor(self._registry, self._cache).run(graph, target, ctx, record.state)
        finally:
            self._settle(record)

    def _emit(self, record: RunRecord, event: RunEvent) -> None:
        """Fold and publish an event the executor did not raise itself."""
        apply_event(record.state, event)
        self.publish(record, event)

    def _settle(self, record: RunRecord) -> None:
        record.done = True
        if record.state.status in (RunStatus.QUEUED, RunStatus.RUNNING):
            record.state.status = RunStatus.ERROR
        if self._store is not None:
            self._store.update(record.state)
        self.publish(record, None)
        self._notify("finished", record)


def _historical(state: RunState) -> RunRecord:
    """A finished run reloaded from the store: snapshot only, no live subscribers."""
    record = RunRecord(state, CancelToken())
    record.done = True
    return record


def _graph_hash(graph: Graph, target: str) -> str:
    nodes = [
        {
            "id": node.id,
            "type": node.type,
            "params": node.params,
            "inputs": {
                port: [[e.from_node, e.output] for e in edges]
                for port, edges in sorted(node.inputs.items())
            },
        }
        for node in graph.nodes
    ]
    payload = json.dumps({"target": target, "nodes": nodes}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
