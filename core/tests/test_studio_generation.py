"""Core-node generation wiring: graph serialization + run-event translation / take saving."""

from __future__ import annotations

import asyncio

from inline_core.studio import moodboard as mb
from inline_core.studio.generation import CoreGeneration
from inline_core.studio.graph_build import build_workflow_graph
from inline_core.studio.store import StudioStore


def _store(tmp_path) -> StudioStore:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("Gen")
    return store


def test_build_workflow_graph_prompt_into_zimage(tmp_path) -> None:
    store = _store(tmp_path)
    conn = store.conn()
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    prompt = mb.add_prompt(conn, 80, 200)
    mb.update_item(conn, prompt["id"], {"data": {"promptText": "a neon city"}})
    mb.create_connector(conn, prompt["id"], z["id"], "out", "prompt")

    graph, target = build_workflow_graph(conn, store.folder(), z["id"])
    assert target == z["id"] and graph["schemaVersion"] == 1
    by_type = {n["type"]: n for n in graph["nodes"]}
    assert by_type["input/text"]["params"] == {"text": "a neon city"}
    zi = by_type["alibaba/z-image-turbo"]
    assert zi["inputs"]["prompt"] == {"from": prompt["id"], "output": "text"}


class _Events:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def broadcast(self, channel: str, payload: dict) -> None:
        self.sent.append((channel, payload))


class _State:
    def __init__(self, takes: list) -> None:
        self.takes = takes
        self.run_id = "run_x"


class _Record:
    def __init__(self, takes: list, done: bool) -> None:
        self.state = _State(takes)
        self.done = done
        self.subscribers: set = set()


class _Take:
    def __init__(self, take_id: str, node_id: str, uri: str, kind: str) -> None:
        self.id = take_id
        self.node_id = node_id
        self.uri = uri
        self.kind = kind


def test_drain_saves_take_sets_output_and_emits_done(tmp_path) -> None:
    store = _store(tmp_path)
    z = mb.add_core_node(store.conn(), "alibaba/z-image-turbo", 0, 0)
    # A produced take: bytes on disk, node_id == the canvas item that made it.
    src = tmp_path / "render.png"
    src.write_bytes(b"\x89PNG image bytes")
    take = _Take("tk1", z["id"], str(src), "image")
    events = _Events()
    gen = CoreGeneration(store, manager=None, events=events)

    asyncio.run(gen._drain(z["id"], _Record([take], done=True)))

    # Done event emitted.
    assert ("events:generationDone", {"targetFrameId": z["id"]}) in events.sent
    # The take was copied into the project's takes/ dir...
    item = mb.get_item(store.conn(), z["id"])
    output = item["data"]["core"]["output"]
    assert output["kind"] == "image" and output["takeId"] == "tk1"
    copied = store.folder() / output["filePath"]
    assert copied.is_file() and copied.read_bytes() == b"\x89PNG image bytes"


def test_drain_translates_error_event(tmp_path) -> None:
    store = _store(tmp_path)
    z = mb.add_core_node(store.conn(), "t", 0, 0)
    events = _Events()
    gen = CoreGeneration(store, manager=None, events=events)
    record = _Record([], done=False)

    async def drive() -> None:
        task = asyncio.create_task(gen._drain(z["id"], record))
        await asyncio.sleep(0)  # let it subscribe
        from inline_core.runtime.progress import ErrorEvent

        for q in list(record.subscribers):
            q.put_nowait(ErrorEvent(run_id="run_x", message="boom", node_id=z["id"]))
        await task

    asyncio.run(drive())
    channels = [c for c, _ in events.sent]
    assert "events:generationError" in channels
    err = next(p for c, p in events.sent if c == "events:generationError")
    assert err["error"] == "boom"
