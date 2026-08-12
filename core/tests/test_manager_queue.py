"""The queue surface on RunManager: observer seam, queue positions, cancel-before-start.

Every run reaches the worker through the manager whoever submitted it, so this is what makes a
run submitted straight to POST /v1/runs visible alongside the ones Studio starts.
"""

from __future__ import annotations

import threading
import time

from inline_core.graph.cache import InMemoryCache
from inline_core.graph.descriptor import NodeDescriptor, Port
from inline_core.graph.registry import Registry
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind, parse_graph
from inline_core.runtime.run import RunStatus
from inline_core.server.manager import RunManager


class _SlowRunner(NodeRunner):
    """Blocks until released, so several runs can be observed queued behind one worker."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def run(self, node, inputs, ctx):  # type: ignore[no-untyped-def]
        self._gate.wait(timeout=5)
        return NodeResult(outputs={"out": "ok"})


class _BoomRunner(NodeRunner):
    def run(self, node, inputs, ctx):  # type: ignore[no-untyped-def]
        raise RuntimeError("runner exploded")


def _registry(runner: NodeRunner) -> Registry:
    registry = Registry()
    registry.register(
        NodeDescriptor(
            type="test/slow",
            title="Slow",
            category="Test",
            outputs=(Port("out", "Out", PortKind.IMAGE),),
        ),
        runner,
    )
    return registry


def _graph() -> object:
    return parse_graph(
        {"schemaVersion": 1, "nodes": [{"id": "n1", "type": "test/slow", "params": {}}]}
    )


def _manager(runner: NodeRunner) -> RunManager:
    return RunManager(_registry(runner), InMemoryCache())


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_observer_sees_queued_then_started_then_finished() -> None:
    gate = threading.Event()
    gate.set()
    manager = _manager(_SlowRunner(gate))
    seen: list[str] = []
    manager.add_observer(lambda phase, _record: seen.append(phase))

    record, _ = manager.submit(_graph(), "n1")
    assert _wait(lambda: "finished" in seen)

    assert seen[0] == "queued"
    assert "started" in seen
    assert seen[-1] == "finished"
    assert record.state.status is RunStatus.DONE


def test_queue_positions_reflect_the_single_worker() -> None:
    gate = threading.Event()
    manager = _manager(_SlowRunner(gate))
    first, _ = manager.submit(_graph(), "n1")
    second, _ = manager.submit(_graph(), "n1")
    third, _ = manager.submit(_graph(), "n1")

    # The first run holds the only worker, so the other two are still in line behind it.
    assert _wait(lambda: first.state.status is RunStatus.RUNNING)
    assert manager.queue_position(second.state.run_id) == 0
    assert manager.queue_position(third.state.run_id) == 1
    # A running run is no longer queued.
    assert manager.queue_position(first.state.run_id) is None

    gate.set()
    assert _wait(lambda: third.done)
    assert manager.queue_position(third.state.run_id) is None


def test_cancelling_a_queued_run_stops_it_before_it_executes() -> None:
    gate = threading.Event()
    manager = _manager(_SlowRunner(gate))
    blocker, _ = manager.submit(_graph(), "n1")
    queued, _ = manager.submit(_graph(), "n1")
    assert _wait(lambda: blocker.state.status is RunStatus.RUNNING)

    assert manager.cancel(queued.state.run_id) is True
    gate.set()

    assert _wait(lambda: queued.done)
    assert queued.state.status is RunStatus.CANCELLED
    # It never ran, so no node ever left the queued state.
    assert all(n.state.value == "queued" for n in queued.state.nodes.values())


def test_a_failing_run_still_settles_and_releases_subscribers() -> None:
    """The terminal sentinel must always be published, or a drain task waits on it forever."""
    manager = _manager(_BoomRunner())
    finished: list[str] = []
    manager.add_observer(lambda phase, _r: finished.append(phase))

    record, _ = manager.submit(_graph(), "n1")
    assert _wait(lambda: record.done)

    assert record.state.status is RunStatus.ERROR
    assert finished[-1] == "finished"


def test_submit_meta_round_trips_untouched() -> None:
    """The engine stores caller metadata but never reads it: that is what keeps it API-agnostic."""
    gate = threading.Event()
    gate.set()
    manager = _manager(_SlowRunner(gate))
    meta = {"projectId": "p1", "itemId": "i1", "anything": [1, 2, 3]}

    record, _ = manager.submit(_graph(), "n1", None, meta)

    assert record.meta == meta
    assert _wait(lambda: record.done)


def test_a_run_with_no_meta_has_an_empty_dict() -> None:
    gate = threading.Event()
    gate.set()
    manager = _manager(_SlowRunner(gate))
    record, _ = manager.submit(_graph(), "n1")
    assert record.meta == {}
