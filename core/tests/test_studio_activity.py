"""The activity registry: project pinning, run tracking, history, and the three run-lifecycle bugs.

The pinning tests are the important ones. A run outlives the project being open, so anything it
writes afterwards has to land in the project it was submitted for.
"""

from __future__ import annotations

import asyncio

from inline_core.studio import moodboard as mb
from inline_core.studio.activity import ActivityRegistry, ActivityRun
from inline_core.studio.generation import CoreGeneration
from inline_core.studio.store import StudioStore, open_project_db


class _Events:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def broadcast(self, channel: str, payload: dict) -> None:
        self.sent.append((channel, payload))

    def channels(self) -> list[str]:
        return [c for c, _ in self.sent]


class _State:
    def __init__(self, takes: list, run_id: str = "run_x") -> None:
        self.takes = takes
        self.run_id = run_id


class _Record:
    def __init__(self, takes: list, done: bool, run_id: str = "run_x") -> None:
        self.state = _State(takes, run_id)
        self.done = done
        self.subscribers: set = set()


class _Take:
    def __init__(self, take_id: str, node_id: str, uri: str, kind: str) -> None:
        self.id = take_id
        self.node_id = node_id
        self.uri = uri
        self.kind = kind


def _store(tmp_path) -> StudioStore:
    return StudioStore(tmp_path / "app", tmp_path / "ws")


def _run(project_id: str = "p1", run_id: str = "r1", status: str = "queued") -> ActivityRun:
    return ActivityRun(
        run_id=run_id,
        kind="generation",
        engine="core",
        origin="studio",
        status=status,  # type: ignore[arg-type]
        title="Z-Image",
        queued_at=1,
        project_id=project_id,
        item_id="i1",
    )


# --- project pinning ----------------------------------------------------------------------------


def test_a_take_lands_in_the_project_the_run_was_submitted_for(tmp_path) -> None:
    """Defect 1: `_save_take` used to resolve the *open* project, so switching mid-run put the
    output in the wrong place."""
    store = _store(tmp_path)
    store.create_project("Alpha")
    alpha = store.project_ref()
    assert alpha is not None
    node = mb.add_core_node(store.conn(), "alibaba/z-image-turbo", 0, 0)
    src = tmp_path / "render.png"
    src.write_bytes(b"\x89PNG bytes")

    # The user switches to another project while the run is still in flight.
    store.create_project("Beta")
    beta = store.project_ref()
    assert beta is not None and beta.folder != alpha.folder

    gen = CoreGeneration(store, manager=None, events=_Events())
    record = _Record([_Take("tk1", node["id"], str(src), "image")], True)
    asyncio.run(gen._drain(node["id"], record, alpha))

    # The bytes went to Alpha, and Beta's takes dir stayed empty.
    assert len(list((alpha.folder / "takes").glob("*.png"))) == 1
    assert list((beta.folder / "takes").glob("*.png")) == []

    # And the canvas node in Alpha's db, not Beta's, points at it.
    conn = open_project_db(alpha.folder)
    item = mb.get_item(conn, node["id"])
    assert (item["data"]["core"]["output"]["takeId"]) == "tk1"
    conn.close()


def test_history_is_written_to_the_pinned_project(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_project("Alpha")
    alpha = store.project_ref()
    assert alpha is not None
    registry = ActivityRegistry(store, _Events())
    registry.track(_run(project_id=alpha.id), ref=alpha)

    store.create_project("Beta")  # switch away before the run finishes
    registry.finish("r1", "done", take_id="tk1")

    conn = open_project_db(alpha.folder)
    rows = conn.execute("SELECT id, status, take_id FROM generation_runs").fetchall()
    conn.close()
    assert [(r["id"], r["status"], r["take_id"]) for r in rows] == [("r1", "done", "tk1")]
    # Beta never saw it.
    assert registry.history() == []


def test_bind_reuses_the_open_connection_for_the_open_project(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_project("Alpha")
    ref = store.project_ref()
    assert ref is not None
    with store.bind(ref) as conn:
        assert conn is store.conn()


# --- run lifecycle bugs -------------------------------------------------------------------------


def test_a_rerun_keeps_its_own_tracking_when_the_previous_drain_exits(tmp_path) -> None:
    """Defect 2: the cancelled run's `finally` used to pop the *successor's* entry."""
    store = _store(tmp_path)
    store.create_project("Alpha")
    ref = store.project_ref()
    node = mb.add_core_node(store.conn(), "alibaba/z-image-turbo", 0, 0)
    gen = CoreGeneration(store, manager=None, events=_Events())

    # The second run has already claimed the slot by the time the first one's drain finishes.
    gen._active[node["id"]] = "run_second"
    asyncio.run(gen._drain(node["id"], _Record([], True, run_id="run_first"), ref))

    assert gen._active.get(node["id"]) == "run_second"


def test_a_cancelled_run_broadcasts_so_the_node_stops_spinning(tmp_path) -> None:
    """Defect 3: a cancel that did not come from this tab used to emit nothing at all."""
    from inline_core.runtime.progress import CancelledEvent

    store = _store(tmp_path)
    store.create_project("Alpha")
    ref = store.project_ref()
    node = mb.add_core_node(store.conn(), "alibaba/z-image-turbo", 0, 0)
    events = _Events()
    gen = CoreGeneration(store, manager=None, events=events)

    record = _Record([], False, run_id="run_c")

    async def drive() -> None:
        task = asyncio.ensure_future(gen._drain(node["id"], record, ref))
        await asyncio.sleep(0)
        for queue in list(record.subscribers):
            queue.put_nowait(CancelledEvent(run_id="run_c"))
        await task

    asyncio.run(drive())

    assert "events:generationCancelled" in events.channels()
    assert "events:generationDone" not in events.channels()


# --- registry behaviour -------------------------------------------------------------------------


def test_live_excludes_finished_runs_and_history_excludes_api_runs(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_project("Alpha")
    ref = store.project_ref()
    assert ref is not None
    registry = ActivityRegistry(store, _Events())

    registry.track(_run(project_id=ref.id, run_id="r1"), ref=ref)
    api = _run(run_id="r2")
    api.origin = "api"
    api.project_id = None
    registry.track(api)  # no ref: an API run belongs to no project

    assert {r.run_id for r in registry.live()} == {"r1", "r2"}

    registry.finish("r1", "done")
    registry.finish("r2", "done")
    assert registry.live() == []
    # Only the project-owned run reached history.
    assert [r["runId"] for r in registry.history()] == ["r1"]


def test_cancel_routes_by_engine_not_by_kind(tmp_path) -> None:
    """A fal run is still `kind=generation`, but it cancels through different machinery."""
    store = _store(tmp_path)
    store.create_project("Alpha")
    registry = ActivityRegistry(store, _Events())
    called: list[str] = []
    registry.set_canceller("core", lambda rid: called.append(f"core:{rid}"))
    registry.set_canceller("fal", lambda rid: called.append(f"fal:{rid}"))

    core_run = _run(run_id="r_core", status="running")
    fal_run = _run(run_id="r_fal", status="running")
    fal_run.engine = "fal"
    registry.track(core_run)
    registry.track(fal_run)

    assert registry.cancel("r_core") is True
    assert registry.cancel("r_fal") is True
    assert registry.cancel("nope") is False
    assert called == ["core:r_core", "fal:r_fal"]


def test_reconcile_settles_rows_left_running_by_a_crash(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_project("Alpha")
    ref = store.project_ref()
    assert ref is not None
    store.conn().execute(
        "INSERT INTO generation_runs (id, project_id, item_id, surface, engine, title, status, "
        "queued_at) VALUES ('stale', ?, 'i1', 'studio', 'core', 'Z-Image', 'running', 1)",
        (ref.id,),
    )

    ActivityRegistry(store, _Events()).reconcile()

    row = store.conn().execute("SELECT status FROM generation_runs WHERE id = 'stale'").fetchone()
    assert row["status"] == "interrupted"


def test_a_status_change_broadcasts_the_whole_live_list(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_project("Alpha")
    events = _Events()
    registry = ActivityRegistry(store, events)

    registry.track(_run(run_id="r1"))
    registry.update("r1", status="running")

    channels = events.channels()
    assert channels == ["events:activityChanged", "events:activityChanged"]
    assert [r["runId"] for r in events.sent[-1][1]["runs"]] == ["r1"]
    assert events.sent[-1][1]["runs"][0]["status"] == "running"
