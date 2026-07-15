"""Serialize a canvas subgraph into an Inline Core graph (schemaVersion 1) — ported from the Studio
``electron/main/core/workflow.ts``. Walks the connector graph upstream from a target node's closure:

- a ``core`` item   -> its Core node type + params (handles are already Core port ids)
- a ``prompt`` item -> an ``input/text`` source node
- an ``asset`` item -> an ``input/image`` source node (local path ref)

Connectors become typed edges (source output port -> target input port). Node ids are the canvas
item ids, so a produced take's ``node_id`` maps straight back to the item that made it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import moodboard as mb


def _source_output_port(source: dict[str, Any] | None, source_handle: str | None) -> str:
    if source and source["type"] == "prompt":
        return "text"
    if source and source["type"] == "asset":
        return "image"
    return source_handle or "out"  # a 'core' item's handles already are Core port ids


def _edges_for(
    item_id: str, connectors: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, str]]:
    inputs: dict[str, dict[str, str]] = {}
    for c in connectors:
        if c["toItemId"] != item_id:
            continue
        data = c.get("data") or {}
        target_handle = data.get("targetHandle") or "in"
        inputs[target_handle] = {
            "from": c["fromItemId"],
            "output": _source_output_port(by_id.get(c["fromItemId"]), data.get("sourceHandle")),
        }
    return inputs


def _item_to_node(
    item: dict[str, Any],
    connectors: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    resolve_asset_path: Callable[[str], str | None],
) -> dict[str, Any] | None:
    data = item.get("data") or {}
    if item["type"] == "core" and data.get("core"):
        return {
            "id": item["id"],
            "type": data["core"]["type"],
            "params": data["core"].get("params") or {},
            "inputs": _edges_for(item["id"], connectors, by_id),
        }
    if item["type"] == "prompt":
        text = data.get("promptText") or ""
        return {"id": item["id"], "type": "input/text", "params": {"text": text}}
    if item["type"] == "asset" and item.get("assetId"):
        path = resolve_asset_path(item["assetId"])
        if not path:
            return None
        return {
            "id": item["id"],
            "type": "input/image",
            "params": {"asset": {"ref": "path", "path": path}},
        }
    return None


def _upstream_closure(target: str, connectors: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    stack = [target]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for c in connectors:
            if c["toItemId"] == node_id and c["fromItemId"] not in seen:
                stack.append(c["fromItemId"])
    return seen


def build_workflow_graph(
    conn: sqlite3.Connection, folder: Path, target_item_id: str
) -> tuple[dict[str, Any], str]:
    """Build the Core graph for a canvas node from the open project's board."""
    board = mb.list_board(conn)
    items, connectors = board["items"], board["connectors"]
    by_id = {i["id"]: i for i in items}

    def resolve_asset_path(asset_id: str) -> str | None:
        row = conn.execute("SELECT file_path FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return str(folder / row["file_path"]) if row else None

    nodes: list[dict[str, Any]] = []
    for node_id in _upstream_closure(target_item_id, connectors):
        item = by_id.get(node_id)
        if item is None:
            continue
        node = _item_to_node(item, connectors, by_id, resolve_asset_path)
        if node is not None:
            nodes.append(node)
    return {"schemaVersion": 1, "nodes": nodes}, target_item_id
