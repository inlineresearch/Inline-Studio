"""The Studio frames/takes/inputs domain port."""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.studio import frames as fr
from inline_core.studio.schema import apply_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute(
        "INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'Proj', 0, 0)"
    )
    return c


def _add_asset(conn: sqlite3.Connection, asset_id: str, kind: str = "image") -> None:
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES (?, 'p', ?, ?, ?, 0)",
        (asset_id, asset_id, f"assets/{asset_id}.png", kind),
    )


def test_add_from_asset_creates_frame_and_input(conn) -> None:
    _add_asset(conn, "a1")
    frame = fr.add_from_asset(conn, "a1")
    assert frame["provider"] == "unset"
    assert frame["inputAssetId"] == "a1"
    inputs = fr.list_inputs(conn)
    assert len(inputs) == 1 and inputs[0]["assetId"] == "a1"
    assert [f["id"] for f in fr.list_frames(conn)] == [frame["id"]]


def test_add_from_missing_asset_raises(conn) -> None:
    with pytest.raises(ValueError):
        fr.add_from_asset(conn, "nope")


def test_create_fal_frame_and_set_params(conn) -> None:
    frame = fr.create_fal_frame(conn, "fal-ai/x", "video", {"steps": 8}, "Cool Model")
    assert frame["provider"] == "fal"
    assert frame["kind"] == "video"
    assert frame["name"] == "Cool Model"
    assert frame["params"] == {"steps": 8}
    updated = fr.set_fal_params(conn, frame["id"], {"steps": 12, "seed": 1})
    assert updated["params"] == {"steps": 12, "seed": 1}


def test_set_model_and_provider(conn) -> None:
    frame = fr.create_empty_frame(conn)
    assert frame["provider"] == "unset"
    to_fal = fr.set_provider(conn, frame["id"], "fal", "m/1", "image", {"a": 1})
    assert to_fal["provider"] == "fal" and to_fal["modelId"] == "m/1" and to_fal["kind"] == "image"
    swapped = fr.set_model(conn, frame["id"], "m/2", "video", {"b": 2})
    assert swapped["modelId"] == "m/2" and swapped["kind"] == "video"
    assert swapped["params"] == {"b": 2}
    to_comfy = fr.set_provider(conn, frame["id"], "comfy")
    assert to_comfy["provider"] == "comfy" and to_comfy["modelId"] is None


def test_rename_reorder_delete(conn) -> None:
    a = fr.create_empty_frame(conn)
    b = fr.create_empty_frame(conn)
    fr.rename_frame(conn, a["id"], "  First  ")
    assert fr.get_frame(conn, a["id"])["name"] == "First"
    with pytest.raises(ValueError):
        fr.rename_frame(conn, a["id"], "   ")
    fr.reorder_frames(conn, [b["id"], a["id"]])
    assert [f["id"] for f in fr.list_frames(conn)] == [b["id"], a["id"]]
    fr.delete_frame(conn, a["id"])
    assert [f["id"] for f in fr.list_frames(conn)] == [b["id"]]


def test_clone_copies_inputs(conn) -> None:
    _add_asset(conn, "a1")
    src = fr.add_from_asset(conn, "a1")
    fr.set_fal_params(conn, src["id"], {"k": 1})
    clone = fr.clone_frame(conn, src["id"])
    assert clone["id"] != src["id"]
    assert clone["name"].endswith("copy")
    clone_inputs = [i for i in fr.list_inputs(conn) if i["frameId"] == clone["id"]]
    assert len(clone_inputs) == 1 and clone_inputs[0]["assetId"] == "a1"


def test_inputs_dedupe_and_source_links(conn) -> None:
    f1 = fr.create_empty_frame(conn)
    f2 = fr.create_empty_frame(conn)
    _add_asset(conn, "a1")
    _add_asset(conn, "a2")
    fr.add_input(conn, f1["id"], "a1")
    assert fr.add_input(conn, f1["id"], "a1")["assetId"] == "a1"  # dedupe: returns existing
    added = fr.add_inputs(conn, f1["id"], ["a1", "a2"])  # a1 skipped
    assert [a["assetId"] for a in added] == ["a2"]
    # Flow link + self-link guard + dedupe.
    link = fr.add_source_input(conn, f1["id"], f2["id"])
    assert link["sourceFrameId"] == f2["id"]
    assert fr.add_source_input(conn, f1["id"], f2["id"])["id"] == link["id"]
    with pytest.raises(ValueError):
        fr.add_source_input(conn, f1["id"], f1["id"])
    fr.remove_input(conn, f1["id"], "a1")
    assert all(i["assetId"] != "a1" for i in fr.list_inputs(conn) if i["frameId"] == f1["id"])


def test_takes_hero_and_delete(conn) -> None:
    frame = fr.create_empty_frame(conn)
    t1 = fr.add_take(conn, frame["id"], "takes/t1.png", "image", {"seed": 1})
    assert fr.get_frame(conn, frame["id"])["heroTakeId"] == t1["id"]  # newest take is hero
    t2 = fr.add_take(conn, frame["id"], "takes/t2.png", "image", {"seed": 2})
    assert fr.get_frame(conn, frame["id"])["heroTakeId"] == t2["id"]
    assert [t["id"] for t in fr.list_takes(conn, frame["id"])] == [t2["id"], t1["id"]]
    assert {t["id"] for t in fr.hero_takes(conn)} == {t2["id"]}
    # Choose t1 as hero, then delete it → hero cleared, path returned.
    fr.set_hero(conn, frame["id"], t1["id"])
    path = fr.delete_take(conn, t1["id"])
    assert path == "takes/t1.png"
    assert fr.get_frame(conn, frame["id"])["heroTakeId"] is None
    assert fr.delete_take(conn, "missing") is None
