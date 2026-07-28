"""Author a reproducible recipe for a generated take: the params it used, its prompt, and the raw
canvas subgraph (items + connectors of the target's upstream closure) so the image can rebuild the
graph that made it when dropped back in - even in someone else's project.

Reuses graph_build's closure walk. Unlike ``build_workflow_graph`` this keeps the *raw moodboard*
items (type / minimal data / position), not the flattened engine graph, because the client rebuilds
canvas nodes from it. Per-item data is trimmed to the rebuild-relevant fields so the recipe stays
small and never nests another node's take history (which itself carries recipes).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import moodboard as mb
from .graph_build import _upstream_closure

RECIPE_VERSION = 1


def _clean_data(item: dict[str, Any]) -> dict[str, Any]:
    """The rebuild-relevant subset of an item's data - never the take history (`core.outputs`)."""
    kind = item["type"]
    data = item.get("data") or {}
    if kind == "core":
        core = data.get("core") or {}
        return {"core": {"type": core.get("type"), "params": core.get("params") or {}}}
    if kind == "prompt":
        return {"promptText": data.get("promptText") or ""}
    if kind == "controlSpace":
        out: dict[str, Any] = {}
        if data.get("controlAssetId"):
            out["controlAssetId"] = data["controlAssetId"]
        if data.get("controlScene"):
            out["controlScene"] = data["controlScene"]
        return out
    if kind == "loader":
        return {"assetIds": data.get("assetIds") or []}
    if kind in ("text", "layer", "director", "trim"):
        return data  # small styling/label data, kept verbatim
    return {}


def _prompt_for(target_id: str, connectors: list[dict[str, Any]], items: dict[str, Any]) -> str:
    """The text of the prompt node wired into the target's ``prompt`` input (for quick display)."""
    for c in connectors:
        if c["toItemId"] == target_id and (c.get("data") or {}).get("targetHandle") == "prompt":
            src = items.get(c["fromItemId"])
            if src and src["type"] == "prompt":
                return str((src.get("data") or {}).get("promptText") or "")
    return ""


def build_recipe(conn: sqlite3.Connection, target_item_id: str) -> dict[str, Any]:
    """The recipe for the take produced by ``target_item_id`` (a canvas item id == node id)."""
    board = mb.list_board(conn)
    items = {i["id"]: i for i in board["items"]}
    connectors = board["connectors"]
    closure = _upstream_closure(target_item_id, connectors)

    recipe_items = [
        {
            "id": item["id"],
            "type": item["type"],
            "data": _clean_data(item),
            "x": item["x"],
            "y": item["y"],
            "width": item["width"],
            "height": item["height"],
            "assetId": item.get("assetId"),
            "frameId": item.get("frameId"),
        }
        for node_id in closure
        if (item := items.get(node_id)) is not None
    ]
    recipe_connectors = [
        {
            "fromItemId": c["fromItemId"],
            "toItemId": c["toItemId"],
            "data": c.get("data") or {},
        }
        for c in connectors
        if c["fromItemId"] in closure and c["toItemId"] in closure
    ]

    target = items.get(target_item_id) or {}
    core = (target.get("data") or {}).get("core") or {}
    return {
        "version": RECIPE_VERSION,
        "app": "inline-studio",
        "target": target_item_id,
        "coreType": core.get("type"),
        "params": core.get("params") or {},
        "prompt": _prompt_for(target_item_id, connectors, items),
        "graph": {"items": recipe_items, "connectors": recipe_connectors},
    }
