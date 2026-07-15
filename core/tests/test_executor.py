from __future__ import annotations

from helpers import build_ctx, build_graph, make_registry
from inline_core.graph.cache import InMemoryCache
from inline_core.graph.executor import Executor
from inline_core.runtime.progress import (
    CollectingEmitter,
    NodeDoneEvent,
    ProgressEvent,
    RunDoneEvent,
)
from inline_core.runtime.run import NodeState, RunState, RunStatus


def test_executor_runs_and_streams() -> None:
    executor = Executor(make_registry(), InMemoryCache())
    emitter = CollectingEmitter()
    state = RunState(run_id="run1", target="m1")

    executor.run(build_graph({"seed": 7}), "m1", build_ctx(emitter, "run1"), state)

    assert any(isinstance(e, ProgressEvent) and e.node_id == "m1" for e in emitter.events)
    assert any(
        isinstance(e, NodeDoneEvent) and e.node_id == "m1" and not e.cached for e in emitter.events
    )
    assert isinstance(emitter.events[-1], RunDoneEvent)
    assert state.status is RunStatus.DONE
    assert state.fraction == 1.0
    assert state.nodes["m1"].state is NodeState.DONE
    assert len(state.takes) == 1


def test_executor_reuses_cache_on_second_run() -> None:
    registry = make_registry()
    cache = InMemoryCache()
    executor = Executor(registry, cache)
    graph = build_graph({"seed": 7})

    ctx = build_ctx(CollectingEmitter(), "run1")
    executor.run(graph, "m1", ctx, RunState(run_id="run1", target="m1"))

    emitter = CollectingEmitter()
    state = RunState(run_id="run2", target="m1")
    executor.run(graph, "m1", build_ctx(emitter, "run2"), state)

    done = [e for e in emitter.events if isinstance(e, NodeDoneEvent) and e.node_id == "m1"]
    assert done and done[0].cached is True
    assert state.nodes["m1"].state is NodeState.CACHED
