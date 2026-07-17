from __future__ import annotations

import time
from pathlib import Path

from helpers import build_graph, make_registry
from inline_core.graph.cache import InMemoryCache
from inline_core.runtime.context import CancelToken
from inline_core.runtime.progress import NodeDoneEvent, Phase, ProgressEvent
from inline_core.runtime.run import RunState, RunStatus
from inline_core.server.manager import RunManager, RunRecord
from inline_core.server.run_store import SqliteRunStore


def _wait(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_run_survives_a_restart(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    first = RunManager(make_registry(), InMemoryCache(), store=SqliteRunStore(db))
    record, _ = first.submit(build_graph({"seed": 7}), "m1")
    _wait(lambda: record.done)
    run_id = record.state.run_id
    first.shutdown()

    # A fresh manager over the same DB: the run and its take are still there.
    second = RunManager(make_registry(), InMemoryCache(), store=SqliteRunStore(db))
    loaded = second.get(run_id)
    assert loaded is not None
    assert loaded.state.status is RunStatus.DONE
    assert loaded.state.nodes["m1"].state.value == "done"
    assert len(loaded.state.takes) == 1
    second.shutdown()


def test_stale_running_run_is_interrupted_on_start(tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    store = SqliteRunStore(db)
    stuck = RunState(run_id="run_stuck", target="m1", status=RunStatus.RUNNING)
    store.create(stuck, None)

    SqliteRunStore(db).interrupt_stale()

    reloaded = SqliteRunStore(db).load("run_stuck")
    assert reloaded is not None
    assert reloaded.status is RunStatus.ERROR
    assert reloaded.error is not None


def test_progress_events_are_coalesced() -> None:
    manager = RunManager(make_registry(), InMemoryCache())
    record = RunRecord(RunState(run_id="r", target="m1"), CancelToken())
    progress = ProgressEvent("r", "m1", Phase.SAMPLE, 0.5)

    assert manager._should_stream(record, progress, 100.0) is True
    assert manager._should_stream(record, progress, 100.05) is False  # within the interval, dropped
    assert manager._should_stream(record, progress, 100.2) is True  # interval elapsed
    # terminal / node_done events are never coalesced
    assert manager._should_stream(record, NodeDoneEvent("r", "m1", False, []), 100.2) is True
    manager.shutdown()
