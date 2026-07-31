"""Several images wired into one list port, in the order the user wired them.

FLUX.2 addresses reference images by position ("the jacket from image 2"), so this ordering is
user-visible semantics rather than an implementation detail. Before this, ``_edges_for`` keyed edges
by target handle and a second wire silently replaced the first.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from inline_core.graph.schema import PortKind
from inline_core.studio import moodboard as mb
from inline_core.studio.graph_build import build_workflow_graph, ref_node_id
from inline_core.studio.store import StudioStore

_FLUX2 = "black-forest-labs/flux-2"


def _list_ports(node_type: str, port_id: str) -> bool:
    """Stands in for the registry: FLUX.2's reference port is the one list port."""
    return node_type == _FLUX2 and port_id == "image"


@pytest.fixture
def store(tmp_path) -> StudioStore:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("Refs")
    return store


def _asset(conn: sqlite3.Connection, asset_id: str) -> str:
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES (?, ?, ?, ?, 'image', 0)",
        (asset_id, mb._project_id(conn), asset_id, f"assets/{asset_id}.png"),
    )
    return asset_id


def _nodes(store: StudioStore, target: str) -> dict[str, dict[str, Any]]:
    graph, _ = build_workflow_graph(store.conn(), store.folder(), target, _list_ports)
    return {n["id"]: n for n in graph["nodes"]}


def test_several_assets_into_one_list_port_keep_wiring_order(store: StudioStore) -> None:
    conn = store.conn()
    flux = mb.add_core_node(conn, _FLUX2, 400, 200)
    items = []
    for name in ("a", "b", "c"):
        _asset(conn, name)
        item = mb.add_asset(conn, name, 0, 0)
        mb.create_connector(conn, item["id"], flux["id"], "out", "image")
        items.append(item["id"])

    edges = _nodes(store, flux["id"])[flux["id"]]["inputs"]["image"]
    assert [e["from"] for e in edges] == items, "all three survive, in wiring order"


def test_a_second_wire_into_a_single_port_still_replaces(store: StudioStore) -> None:
    # Long-standing behaviour on non-list ports: the canvas treats a second wire as a replacement.
    # Turning that into a validation error would break existing boards.
    conn = store.conn()
    flux = mb.add_core_node(conn, _FLUX2, 400, 200)
    first = mb.add_prompt(conn, 0, 0)
    second = mb.add_prompt(conn, 0, 100)
    mb.create_connector(conn, first["id"], flux["id"], "out", "prompt")
    mb.create_connector(conn, second["id"], flux["id"], "out", "prompt")

    edges = _nodes(store, flux["id"])[flux["id"]]["inputs"]["prompt"]
    assert [e["from"] for e in edges] == [second["id"]]


def test_a_load_assets_node_contributes_every_asset_to_a_list_port(store: StudioStore) -> None:
    conn = store.conn()
    for name in ("a", "b", "c"):
        _asset(conn, name)
    flux = mb.add_core_node(conn, _FLUX2, 400, 200)
    loader = mb.add_loader(conn, 80, 200)
    mb.update_item(conn, loader["id"], {"data": {"assetIds": ["a", "b", "c"]}})
    mb.create_connector(conn, loader["id"], flux["id"], "out", "image")

    nodes = _nodes(store, flux["id"])
    edges = nodes[flux["id"]]["inputs"]["image"]
    assert [e["from"] for e in edges] == [ref_node_id(loader["id"], i) for i in range(3)]
    # One frozen image source per asset, and the un-fanned loader node is gone (no dangling edge).
    assert loader["id"] not in nodes
    paths = [nodes[e["from"]]["params"]["asset"]["path"] for e in edges]
    assert paths == [str(store.folder() / f"assets/{n}.png") for n in ("a", "b", "c")]


def test_a_load_assets_node_still_feeds_its_hero_asset_to_a_single_port(store: StudioStore) -> None:
    # Z-Image's image port is not a list, so the loader keeps its hero-asset behaviour exactly.
    conn = store.conn()
    for name in ("a", "b"):
        _asset(conn, name)
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    loader = mb.add_loader(conn, 80, 200)
    mb.update_item(conn, loader["id"], {"data": {"assetIds": ["a", "b"]}})
    mb.create_connector(conn, loader["id"], z["id"], "out", "image")

    nodes = _nodes(store, z["id"])
    assert nodes[z["id"]]["inputs"]["image"] == [{"from": loader["id"], "output": "image"}]
    assert nodes[loader["id"]]["params"]["asset"]["path"] == str(store.folder() / "assets/a.png")


def test_a_single_asset_loader_is_not_fanned_out(store: StudioStore) -> None:
    conn = store.conn()
    _asset(conn, "a")
    flux = mb.add_core_node(conn, _FLUX2, 400, 200)
    loader = mb.add_loader(conn, 80, 200)
    mb.update_item(conn, loader["id"], {"data": {"assetIds": ["a"]}})
    mb.create_connector(conn, loader["id"], flux["id"], "out", "image")

    nodes = _nodes(store, flux["id"])
    assert nodes[flux["id"]]["inputs"]["image"] == [{"from": loader["id"], "output": "image"}]
    assert loader["id"] in nodes


def test_a_structure_map_rides_its_own_port_alongside_references(store: StudioStore) -> None:
    conn = store.conn()
    _asset(conn, "ref")
    _asset(conn, "pose")
    flux = mb.add_core_node(conn, _FLUX2, 400, 200)
    ref = mb.add_asset(conn, "ref", 0, 0)
    control = mb.add_control_space(conn, 0, 100)
    mb.update_item(conn, control["id"], {"data": {"controlAssetId": "pose"}})
    mb.create_connector(conn, ref["id"], flux["id"], "out", "image")
    mb.create_connector(conn, control["id"], flux["id"], "out", "control_image")

    inputs = _nodes(store, flux["id"])[flux["id"]]["inputs"]
    assert [e["from"] for e in inputs["image"]] == [ref["id"]]
    assert [e["from"] for e in inputs["control_image"]] == [control["id"]]


def test_without_a_resolver_every_port_stays_single_valued(store: StudioStore) -> None:
    # The pre-multi-reference behaviour, which is what a caller with no registry gets.
    conn = store.conn()
    flux = mb.add_core_node(conn, _FLUX2, 400, 200)
    ids = []
    for name in ("a", "b"):
        _asset(conn, name)
        item = mb.add_asset(conn, name, 0, 0)
        mb.create_connector(conn, item["id"], flux["id"], "out", "image")
        ids.append(item["id"])

    graph, _ = build_workflow_graph(conn, store.folder(), flux["id"])
    node = next(n for n in graph["nodes"] if n["id"] == flux["id"])
    assert [e["from"] for e in node["inputs"]["image"]] == [ids[-1]]


def test_the_graph_validates_and_only_the_list_port_takes_many_edges() -> None:
    """The engine's own rule: >1 edge is rejected everywhere except a list port."""
    from inline_core.models.flux2.runner import FLUX2

    image = FLUX2.input("image")
    assert image is not None and image.kind is PortKind.IMAGE_LIST
    for port in FLUX2.inputs:
        if port.id != "image":
            assert port.kind is not PortKind.IMAGE_LIST, f"{port.id} would silently accept fan-in"
