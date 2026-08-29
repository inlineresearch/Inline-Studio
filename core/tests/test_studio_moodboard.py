"""The Studio moodboard domain port: items, connectors, board replace, prompt resolution."""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.studio import moodboard as mb
from inline_core.studio.schema import apply_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'Proj', 0, 0)")
    return c


def test_add_various_items(conn) -> None:
    text = mb.add_text(conn, 10, 20)
    assert text["type"] == "text" and text["data"]["text"]["text"] == "Text"
    core = mb.add_core_node(conn, "alibaba/z-image-turbo", 0, 0)
    assert core["type"] == "core" and core["data"]["core"]["type"] == "alibaba/z-image-turbo"
    prompt = mb.add_prompt(conn, 0, 0)
    assert prompt["type"] == "prompt" and prompt["data"]["promptText"] == ""
    # z-index increments; layers pin to 0.
    assert core["zIndex"] > text["zIndex"]
    assert mb.add_layer(conn, 0, 0)["zIndex"] == 0
    assert len(mb.list_items(conn)) == 4


def test_add_frame_from_asset_creates_frame_and_item(conn) -> None:
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('a1', 'p', 'a', 'assets/a.png', 'image', 0)"
    )
    item = mb.add_frame_from_asset(conn, "a1", 5, 5)
    assert item["type"] == "frame" and item["frameId"]
    assert conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 1


def test_add_gen_node_uses_caller_metadata(conn) -> None:
    item = mb.add_gen_node(conn, "fal/x", 0, 0, kind="video", params={"steps": 8}, title="X")
    assert item["type"] == "frame" and item["width"] == 240
    frame = conn.execute(
        "SELECT provider, kind FROM frames WHERE id = ?", (item["frameId"],)
    ).fetchone()
    assert frame["provider"] == "fal" and frame["kind"] == "video"


def test_update_item_geometry_and_data(conn) -> None:
    item = mb.add_prompt(conn, 0, 0)
    updated = mb.update_item(conn, item["id"], {"x": 100, "y": 50, "data": {"promptText": "hi"}})
    assert updated["x"] == 100 and updated["y"] == 50
    assert updated["data"]["promptText"] == "hi"
    # Booleans must not be treated as numbers for geometry.
    again = mb.update_item(conn, item["id"], {"width": 300})
    assert again["width"] == 300


def test_connectors_and_delete_cascade(conn) -> None:
    a = mb.add_prompt(conn, 0, 0)
    b = mb.add_core_node(conn, "t", 0, 0)
    conn_row = mb.create_connector(conn, a["id"], b["id"], "out", "prompt")
    assert conn_row["data"]["targetHandle"] == "prompt"
    assert len(mb.list_board(conn)["connectors"]) == 1
    mb.set_connector_volume(conn, conn_row["id"], 2.0)  # clamped to 1
    stored = conn.execute(
        "SELECT data FROM moodboard_connectors WHERE id = ?", (conn_row["id"],)
    ).fetchone()
    import json

    assert json.loads(stored["data"])["volume"] == 1.0
    # Deleting an item removes its connectors.
    mb.delete_item(conn, b["id"])
    assert mb.list_board(conn)["connectors"] == []


def test_replace_board_roundtrip(conn) -> None:
    a = mb.add_prompt(conn, 1, 1)
    b = mb.add_core_node(conn, "t", 2, 2)
    mb.create_connector(conn, a["id"], b["id"])
    snapshot = mb.list_board(conn)
    mb.replace_board(conn, [], [])
    assert mb.list_board(conn) == {"items": [], "connectors": []}
    mb.replace_board(conn, snapshot["items"], snapshot["connectors"])
    restored = mb.list_board(conn)
    assert {i["id"] for i in restored["items"]} == {a["id"], b["id"]}
    assert len(restored["connectors"]) == 1


def test_prompt_text_for_frame(conn) -> None:
    prompt = mb.add_prompt(conn, 0, 0)
    mb.update_item(conn, prompt["id"], {"data": {"promptText": "a neon city"}})
    gen = mb.add_gen_node(conn, "fal/x", 0, 0, kind="image", params={}, title="X")
    assert mb.prompt_text_for_frame(conn, gen["frameId"]) is None  # not wired yet
    mb.create_connector(conn, prompt["id"], gen["id"], "out", "prompt")
    assert mb.prompt_text_for_frame(conn, gen["frameId"]) == "a neon city"


def test_a_client_write_cannot_drop_a_take_it_had_not_loaded(conn) -> None:
    """`data` is one JSON blob replaced wholesale, so a browser patch carries whatever that tab last
    loaded. A render landing between its load and its write used to be erased by it: the file stayed
    on disk and `core.outputs`, its only record, was gone. Cost two finished videos."""
    iid = mb.add_core_node(conn, "minimax/h3-reference-to-video", 0, 0)["id"]
    stale = mb.get_item(conn, iid)["data"]  # what a tab loaded before the render

    mb.set_core_node_output(
        conn, iid, {"takeId": "landed", "filePath": "takes/x.mp4", "kind": "video", "createdAt": 20}
    )

    # The tab now writes back its own copy, which has never seen "landed".
    patch = {"data": {**stale, "core": {**stale["core"], "params": {"steps": 4}}}}
    mb.client_update_item(conn, iid, patch)

    core = mb.get_item(conn, iid)["data"]["core"]
    assert [o["takeId"] for o in core["outputs"]] == ["landed"], "the take must survive the write"
    assert core["params"] == {"steps": 4}, "the client's own edit must still land"


def test_a_client_can_still_choose_which_take_is_active(conn) -> None:
    """Only entries the client could not see are restored, so selecting a take keeps working."""
    iid = mb.add_core_node(conn, "minimax/h3-reference-to-video", 0, 0)["id"]
    for n, at in (("a", 10), ("b", 20)):
        mb.set_core_node_output(
            conn, iid, {"takeId": n, "filePath": f"takes/{n}.mp4", "kind": "video", "createdAt": at}
        )
    data = mb.get_item(conn, iid)["data"]
    older = next(o for o in data["core"]["outputs"] if o["takeId"] == "a")

    mb.client_update_item(conn, iid, {"data": {**data, "core": {**data["core"], "output": older}}})

    core = mb.get_item(conn, iid)["data"]["core"]
    assert core["output"]["takeId"] == "a"
    assert {o["takeId"] for o in core["outputs"]} == {"a", "b"}
