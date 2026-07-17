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
from concurrent.futures import ThreadPoolExecutor
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
from ..runtime.progress import ProgressEmitter, ProgressEvent, RunEvent
from ..runtime.run import NodeRuntimeState, RunState
from ..takes import Take
from .run_store import RunStore

# ~10 progress events per second per run to the stream (contract section 6). The snapshot stays
# authoritative, so dropping intermediate ticks only affects stream chattiness.
_MIN_PROGRESS_INTERVAL = 0.1


class RunConflict(InlineCoreError):
    """A clientRunId was reused with a different graph."""


class RunRecord:
    def __init__(self, state: RunState, cancel: CancelToken) -> None:
        self.state = state
        self.cancel = cancel
        self.subscribers: set[asyncio.Queue[RunEvent | None]] = set()
        self.done = False
        self.last_progress = 0.0


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
        if store is not None:
            store.interrupt_stale()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def shutdown(self) -> None:
        for record in list(self._runs.values()):
            record.cancel.cancel()
        self._pool.shutdown(wait=False)

    def submit(
        self, graph: Graph, target: str, client_run_id: str | None = None
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
        record = RunRecord(state, CancelToken())
        with self._lock:
            self._runs[run_id] = record
            if client_run_id is not None:
                self._by_client[client_run_id] = run_id
                self._graph_hash[run_id] = graph_hash
        if self._store is not None:
            self._store.create(state, client_run_id)
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
        loop = self._loop
        if loop is None:
            return
        if not self._should_stream(record, event, time.monotonic()):
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
        ctx = ExecutionContext(
            run_id=record.state.run_id,
            policy=self._policy,
            emitter=_BroadcastEmitter(self, record),
            cancel=record.cancel,
        )
        Executor(self._registry, self._cache).run(graph, target, ctx, record.state)
        record.done = True
        if self._store is not None:
            self._store.update(record.state)
        self.publish(record, None)


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
