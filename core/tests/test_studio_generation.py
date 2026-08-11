"""Core-node generation wiring: graph serialization + run-event translation / take saving."""

from __future__ import annotations

import asyncio

from inline_core.studio import frames as fr
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
    assert zi["inputs"]["prompt"] == [{"from": prompt["id"], "output": "text"}]


def test_build_workflow_graph_frame_output_into_zimage_image(tmp_path) -> None:
    """A rendered frame wired into Z-Image's image port becomes an input/image source node pointing
    at the frame's hero take - no dangling edge to a non-emitted node."""
    store = _store(tmp_path)
    conn = store.conn()
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    frame_item = mb.add_empty_frame(conn, 80, 200)
    take = fr.add_take(conn, frame_item["frameId"], "takes/hero.png", "image", {})
    mb.create_connector(conn, frame_item["id"], z["id"], "out", "image")

    graph, _ = build_workflow_graph(conn, store.folder(), z["id"])
    by_id = {n["id"]: n for n in graph["nodes"]}
    # The frame item is emitted as an input/image node pointing at its hero take's absolute path.
    frame_node = by_id[frame_item["id"]]
    assert frame_node["type"] == "input/image"
    assert frame_node["params"]["asset"]["path"] == str(store.folder() / "takes/hero.png")
    # Z-Image's image input references the frame node's "image" output - no dangling edge.
    zi = by_id[z["id"]]
    assert zi["inputs"]["image"] == [{"from": frame_item["id"], "output": "image"}]
    assert take["id"]  # hero take exists


def test_build_workflow_graph_loader_into_zimage_image(tmp_path) -> None:
    """A standalone Load Assets loader (its hero asset) wired into Z-Image becomes an input/image
    source node - no frame involved."""
    store = _store(tmp_path)
    conn = store.conn()
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('a1', ?, 'a', 'assets/a.png', 'image', 0)",
        (mb._project_id(conn),),
    )
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    loader = mb.add_loader(conn, 80, 200)
    assert loader["type"] == "loader" and loader["data"]["assetIds"] == []
    mb.update_item(conn, loader["id"], {"data": {"assetIds": ["a1"]}})
    mb.create_connector(conn, loader["id"], z["id"], "out", "image")

    graph, _ = build_workflow_graph(conn, store.folder(), z["id"])
    by_id = {n["id"]: n for n in graph["nodes"]}
    loader_node = by_id[loader["id"]]
    assert loader_node["type"] == "input/image"
    assert loader_node["params"]["asset"]["path"] == str(store.folder() / "assets/a.png")
    assert by_id[z["id"]]["inputs"]["image"] == [{"from": loader["id"], "output": "image"}]


def test_build_workflow_graph_control_space_into_zimage_control(tmp_path) -> None:
    """A Control Space node's rendered OpenPose map, wired into Z-Image's control_image port,
    becomes an input/image source node targeting that named input."""
    store = _store(tmp_path)
    conn = store.conn()
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('pose1', ?, 'pose', 'assets/pose.png', 'image', 0)",
        (mb._project_id(conn),),
    )
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    cs = mb.add_control_space(conn, 80, 200)
    assert cs["type"] == "controlSpace"
    mb.update_item(conn, cs["id"], {"data": {"controlAssetId": "pose1"}})
    mb.create_connector(conn, cs["id"], z["id"], "out", "control_image")

    graph, _ = build_workflow_graph(conn, store.folder(), z["id"])
    by_id = {n["id"]: n for n in graph["nodes"]}
    cs_node = by_id[cs["id"]]
    assert cs_node["type"] == "input/image"
    assert cs_node["params"]["asset"]["path"] == str(store.folder() / "assets/pose.png")
    # The map lands on the control_image input, not the plain image input.
    assert by_id[z["id"]]["inputs"]["control_image"] == [{"from": cs["id"], "output": "image"}]


def test_build_workflow_graph_control_space_without_render_is_dropped(tmp_path) -> None:
    """A Control Space node with no rendered map yet emits no source node (no dangling edge)."""
    store = _store(tmp_path)
    conn = store.conn()
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    cs = mb.add_control_space(conn, 80, 200)
    mb.create_connector(conn, cs["id"], z["id"], "out", "control_image")

    graph, _ = build_workflow_graph(conn, store.folder(), z["id"])
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert cs["id"] not in by_id


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


# --- multi-take nodes: which take claims the node's canvas output --------------------------------


class _Registry:
    """Just enough registry for `_is_primary_output`: a type and the media kind it declares."""

    def __init__(self, node_type: str, output_kind) -> None:  # type: ignore[no-untyped-def]
        self._type = node_type
        self._kind = output_kind

    def has(self, node_type: str) -> bool:
        return node_type == self._type

    def get(self, node_type: str):  # type: ignore[no-untyped-def]
        from inline_core.graph.descriptor import NodeDescriptor

        return NodeDescriptor(
            type=self._type, title="X", category="Generate", output_kind=self._kind
        )


def _video_and_audio(tmp_path, item_id: str):  # type: ignore[no-untyped-def]
    video_src = tmp_path / "clip.mp4"
    video_src.write_bytes(b"mp4 bytes")
    audio_src = tmp_path / "clip.wav"
    audio_src.write_bytes(b"wav bytes")
    return [
        _Take("tk_v", item_id, str(video_src), "video"),
        _Take("tk_a", item_id, str(audio_src), "audio"),
    ]


def test_both_takes_persist_but_only_the_declared_kind_claims_the_card(tmp_path) -> None:
    """H3 returns the muxed video and its soundtrack. Both must land in the project, and the video
    must stay what the card shows - before the gate, the audio take saved last and won the slot."""
    from inline_core.media import MediaKind

    store = _store(tmp_path)
    node = mb.add_core_node(store.conn(), "minimax/h3-text-to-video", 0, 0)
    takes = _video_and_audio(tmp_path, node["id"])
    gen = CoreGeneration(
        store, manager=None, events=_Events(),
        registry=_Registry("minimax/h3-text-to-video", MediaKind.VIDEO),
    )

    asyncio.run(gen._drain(node["id"], _Record(takes, done=True)))

    output = mb.get_item(store.conn(), node["id"])["data"]["core"]["output"]
    assert output["kind"] == "video" and output["takeId"] == "tk_v"
    # Both takes were copied in, so the soundtrack survives the run rather than only existing in it.
    written = sorted(p.suffix for p in (store.folder() / "takes").iterdir())
    assert written == [".mp4", ".wav"]


def test_without_a_registry_the_last_take_still_wins(tmp_path) -> None:
    """The gate must be inert when the answer is unknowable, or a torch-less install regresses."""
    store = _store(tmp_path)
    node = mb.add_core_node(store.conn(), "minimax/h3-text-to-video", 0, 0)
    takes = _video_and_audio(tmp_path, node["id"])
    gen = CoreGeneration(store, manager=None, events=_Events())

    asyncio.run(gen._drain(node["id"], _Record(takes, done=True)))

    output = mb.get_item(store.conn(), node["id"])["data"]["core"]["output"]
    assert output["takeId"] == "tk_a"  # unchanged pre-gate behaviour


def test_single_take_image_node_is_unaffected_by_the_gate(tmp_path) -> None:
    from inline_core.media import MediaKind

    store = _store(tmp_path)
    z = mb.add_core_node(store.conn(), "alibaba/z-image-turbo", 0, 0)
    src = tmp_path / "render.png"
    src.write_bytes(b"\x89PNG image bytes")
    gen = CoreGeneration(
        store, manager=None, events=_Events(),
        registry=_Registry("alibaba/z-image-turbo", MediaKind.IMAGE),
    )

    asyncio.run(gen._drain(z["id"], _Record([_Take("tk1", z["id"], str(src), "image")], done=True)))

    output = mb.get_item(store.conn(), z["id"])["data"]["core"]["output"]
    assert output["kind"] == "image" and output["takeId"] == "tk1"


# --- surviving a page refresh --------------------------------------------------------------------


class _Events:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def broadcast(self, channel: str, payload: dict) -> None:
        self.sent.append((channel, payload))


def test_active_reports_runs_in_flight_with_their_last_progress() -> None:
    """A refresh throws away the client's queue while the run carries on. Without this the UI shows
    an empty queue against a GPU that is still working."""
    gen = CoreGeneration(store=None, manager=None, events=_Events())
    gen._active["item-1"] = "run-1"
    gen._progress("item-1", 0.4, "Sampling")

    assert gen.active() == [{"frameId": "item-1", "fraction": 0.4, "status": "Sampling"}]


def test_a_run_that_has_not_reported_yet_still_appears() -> None:
    """H3 emits nothing for minutes while the base loads, which is exactly when someone refreshes,
    so an entry with no progress yet has to come back rather than be omitted."""
    gen = CoreGeneration(store=None, manager=None, events=_Events())
    gen._active["item-2"] = "run-2"

    assert gen.active() == [{"frameId": "item-2", "fraction": None, "status": None}]


def test_a_finished_run_is_not_reported() -> None:
    gen = CoreGeneration(store=None, manager=None, events=_Events())
    gen._active["item-3"] = "run-3"
    gen._progress("item-3", 0.9, "Decoding")
    gen._active.pop("item-3")
    gen._last.pop("item-3")

    assert gen.active() == []
