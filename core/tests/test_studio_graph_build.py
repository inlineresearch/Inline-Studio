"""Canvas -> Core graph serialization, and the Control Space facing hint folded into the prompt."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from inline_core.studio import moodboard as mb
from inline_core.studio.graph_build import build_workflow_graph
from inline_core.studio.schema import apply_schema

BACK_HINT = {
    "positive": "seen from behind, back view, back of the head",
    "negative": "face visible, looking at the camera",
}


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'Proj', 0, 0)")
    c.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('a1', 'p', 'map', 'assets/map.png', 'image', 0)"
    )
    return c


def _board(
    conn: sqlite3.Connection, scene: dict[str, Any] | None, prompt_text: str = "a woman standing"
) -> tuple[str, str]:
    """A prompt + a Control Space map wired into a gen node. Returns (gen id, prompt id)."""
    control = mb.add_control_space(conn, 0, 0)
    mb.update_item(conn, control["id"], {"data": {"controlAssetId": "a1", "controlScene": scene}})
    prompt = mb.add_prompt(conn, 0, 0)
    mb.update_item(conn, prompt["id"], {"data": {"promptText": prompt_text}})
    gen = mb.add_core_node(conn, "alibaba/z-image-turbo", 0, 0)
    mb.create_connector(conn, prompt["id"], gen["id"], "text", "prompt")
    mb.create_connector(conn, control["id"], gen["id"], "image", "control_image")
    return gen["id"], prompt["id"]


def _nodes(conn: sqlite3.Connection, target: str) -> dict[str, dict[str, Any]]:
    graph, _ = build_workflow_graph(conn, Path("/tmp"), target)
    return {n["id"]: n for n in graph["nodes"]}


def test_back_facing_control_space_states_the_facing_in_the_prompt(conn) -> None:
    gen_id, prompt_id = _board(conn, {"facing": ["back"], "promptHint": BACK_HINT})
    nodes = _nodes(conn, gen_id)

    # The prompt is rewired to a derived text node; the shared prompt node is left untouched.
    source = nodes[gen_id]["inputs"]["prompt"][0]["from"]
    assert source != prompt_id
    assert nodes[source]["params"]["text"] == f"a woman standing, {BACK_HINT['positive']}"
    assert nodes[prompt_id]["params"]["text"] == "a woman standing"
    assert nodes[gen_id]["params"]["negative_prompt"] == BACK_HINT["negative"]


def test_the_hint_appends_to_an_existing_negative_prompt(conn) -> None:
    gen_id, _ = _board(conn, {"facing": ["back"], "promptHint": BACK_HINT})
    item_id = next(i["id"] for i in mb.list_items(conn) if i["type"] == "core")
    core = {"type": "alibaba/z-image-turbo", "params": {"negative_prompt": "blur"}}
    mb.update_item(conn, item_id, {"data": {"core": core}})
    negative = _nodes(conn, gen_id)[gen_id]["params"]["negative_prompt"]
    assert negative == f"blur, {BACK_HINT['negative']}"


def test_the_hint_never_stacks_when_a_restored_take_already_carries_it(conn) -> None:
    """Restoring a take writes the injected params onto the node; re-running must not re-add."""
    gen_id, prompt_id = _board(conn, {"facing": ["back"], "promptHint": BACK_HINT})
    item_id = next(i["id"] for i in mb.list_items(conn) if i["type"] == "core")
    core = {"type": "krea/krea-2-turbo", "params": {"negative_prompt": BACK_HINT["negative"]}}
    mb.update_item(conn, item_id, {"data": {"core": core}})
    mb.update_item(
        conn, prompt_id, {"data": {"promptText": f"a woman standing, {BACK_HINT['positive']}"}}
    )
    nodes = _nodes(conn, gen_id)
    assert nodes[gen_id]["params"]["negative_prompt"] == BACK_HINT["negative"]
    assert nodes[gen_id]["inputs"]["prompt"][0]["from"] == prompt_id  # no second derived node


@pytest.mark.parametrize(
    "scene",
    [
        {"facing": ["front"], "promptHint": None},
        {"facing": ["back"], "promptHint": BACK_HINT, "applyPromptHint": False},
        {"facing": ["back", "front"]},
        None,
    ],
)
def test_no_hint_leaves_the_prompt_alone(conn, scene) -> None:
    gen_id, prompt_id = _board(conn, scene)
    nodes = _nodes(conn, gen_id)
    assert nodes[gen_id]["inputs"]["prompt"][0]["from"] == prompt_id
    assert not nodes[gen_id]["params"].get("negative_prompt")


def test_an_unresolvable_control_map_adds_nothing(conn) -> None:
    """No control image in the graph means no ControlNet ran - the facing claim would be a lie."""
    gen_id, prompt_id = _board(conn, {"facing": ["back"], "promptHint": BACK_HINT})
    control_id = next(i["id"] for i in mb.list_items(conn) if i["type"] == "controlSpace")
    mb.update_item(conn, control_id, {"data": {"controlScene": {"promptHint": BACK_HINT}}})
    nodes = _nodes(conn, gen_id)
    assert nodes[gen_id]["inputs"]["prompt"][0]["from"] == prompt_id
    assert not nodes[gen_id]["params"].get("negative_prompt")


def test_an_asset_this_project_does_not_have_takes_its_edge_with_it(conn) -> None:
    """A board exported from one project and imported into another carries asset ids the new project
    has never seen. The source node is skipped, and the edge naming it has to go too - left in, the
    run failed validation with a bare canvas id ("No node with id '6e29da67...'")."""
    loader = mb.add_loader(conn, 0, 0)
    mb.update_item(conn, loader["id"], {"data": {"assetIds": ["from-another-project"]}})
    gen = mb.add_core_node(conn, "alibaba/z-image-turbo", 0, 0)
    mb.create_connector(conn, loader["id"], gen["id"], "image", "image")

    nodes = _nodes(conn, gen["id"])
    assert loader["id"] not in nodes
    assert "image" not in nodes[gen["id"]]["inputs"]


def test_one_unresolvable_asset_does_not_dangle_the_rest_of_a_fanned_out_loader(conn) -> None:
    """A Load Assets node on a list port emits one source per asset and skips the ones that do not
    resolve, so the numbered edges have to be pruned to match or the run dies on the gap."""
    loader = mb.add_loader(conn, 0, 0)
    mb.update_item(conn, loader["id"], {"data": {"assetIds": ["a1", "gone", "a1"]}})
    gen = mb.add_core_node(conn, "black-forest-labs/flux-2", 0, 0)
    mb.create_connector(conn, loader["id"], gen["id"], "image", "images")

    graph, _ = build_workflow_graph(
        conn, Path("/tmp"), gen["id"], lambda _type, port: port == "images"
    )
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = nodes[gen["id"]]["inputs"]["images"]
    assert [e["from"] for e in edges] == [n["id"] for n in graph["nodes"] if "::ref" in n["id"]]


def test_a_wired_clip_becomes_a_video_source_not_an_image_one(conn) -> None:
    """Every asset became `input/image` whatever it held, so wiring a clip into a video port failed
    validation with "Cannot wire image into video" about a file that was already a video.

    The node's type and its edge's output port are asserted together. They are one contract: typing
    the node alone would leave the edge naming a port the node does not have.
    """
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('v1', 'p', 'ref', 'assets/ref.mp4', 'video', 0)"
    )
    clip = mb.add_asset(conn, "v1", 0, 0)
    gen = mb.add_core_node(conn, "lightricks/ltx-2-5-text-to-video", 0, 0)
    mb.create_connector(conn, clip["id"], gen["id"], "image", "reference")

    nodes = _nodes(conn, gen["id"])
    assert nodes[clip["id"]]["type"] == "input/video"
    assert nodes[gen["id"]]["inputs"]["reference"][0]["output"] == "video"
