"""Serialize a canvas subgraph into an Inline Core graph (schemaVersion 1) - ported from the Studio
``electron/main/core/workflow.ts``. Walks the connector graph upstream from a target node's closure:

- a ``core`` item   -> its Core node type + params (handles are already Core port ids)
- a ``prompt`` item -> an ``input/text`` source node
- an ``asset`` item -> an ``input/image`` or ``input/video`` source node (local path ref), by kind
- a ``frame`` item  -> the same, pointing at the frame's resolved output file (its hero take), so
  wiring a rendered frame into a Core node feeds that media without recomputing the frame. This is
  the closure boundary: upstream frames are frozen curated inputs, not re-run.

Connectors become typed edges (source output port -> target input port). Node ids are the canvas
item ids, so a produced take's ``node_id`` maps straight back to the item that made it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import frames as fr
from . import moodboard as mb

# A media source node is picked by the asset/take kind; anything we don't generate for reads as an
# image, which is what every Core node consumed before video existed.
_MEDIA_NODES = {"video": ("input/video", "video")}
_MEDIA_DEFAULT = ("input/image", "image")


def _edges_for(
    item_id: str, connectors: list[dict[str, Any]], output_port: dict[str, str]
) -> dict[str, dict[str, str]]:
    inputs: dict[str, dict[str, str]] = {}
    for c in connectors:
        if c["toItemId"] != item_id:
            continue
        data = c.get("data") or {}
        target_handle = data.get("targetHandle") or "in"
        source = c["fromItemId"]
        inputs[target_handle] = {
            "from": source,
            # A 'core' item isn't in the map - its handles already are Core port ids.
            "output": output_port.get(source) or data.get("sourceHandle") or "out",
        }
    return inputs


def _source_node(
    item: dict[str, Any],
    resolve_asset: Callable[[str], tuple[str, str] | None],
    resolve_frame: Callable[[str], tuple[str, str] | None],
) -> tuple[dict[str, Any], str] | None:
    """A non-``core`` item as (node, its output port), or None when it resolves to no file."""
    data = item.get("data") or {}
    if item["type"] == "prompt":
        text = data.get("promptText") or ""
        return {"id": item["id"], "type": "input/text", "params": {"text": text}}, "text"

    resolved = None
    if item["type"] == "asset" and item.get("assetId"):
        resolved = resolve_asset(item["assetId"])
    elif item["type"] == "frame" and item.get("frameId"):
        resolved = resolve_frame(item["frameId"])
    if resolved is None:
        return None

    path, kind = resolved
    node_type, port = _MEDIA_NODES.get(kind, _MEDIA_DEFAULT)
    node = {
        "id": item["id"],
        "type": node_type,
        "params": {"asset": {"ref": "path", "path": path}},
    }
    return node, port


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

    def resolve_asset(asset_id: str) -> tuple[str, str] | None:
        row = conn.execute(
            "SELECT file_path, kind FROM assets WHERE id = ?", (asset_id,)
        ).fetchone()
        return (str(folder / row["file_path"]), row["kind"]) if row else None

    def resolve_frame(frame_id: str) -> tuple[str, str] | None:
        out = fr.resolve_frame_file(conn, frame_id)
        return (str(folder / out["filePath"]), out["kind"]) if out else None

    closure = _upstream_closure(target_item_id, connectors)
    # Source nodes first: a 'core' item's edges need to know which port each source emits on, and
    # that now depends on the resolved media kind rather than the item type alone.
    sources: list[dict[str, Any]] = []
    output_port: dict[str, str] = {}
    core_items: list[dict[str, Any]] = []
    for node_id in closure:
        item = by_id.get(node_id)
        if item is None:
            continue
        if item["type"] == "core" and (item.get("data") or {}).get("core"):
            core_items.append(item)
            continue
        resolved = _source_node(item, resolve_asset, resolve_frame)
        if resolved is not None:
            node, port = resolved
            sources.append(node)
            output_port[item["id"]] = port

    nodes = sources + [
        {
            "id": item["id"],
            "type": item["data"]["core"]["type"],
            "params": item["data"]["core"].get("params") or {},
            "inputs": _edges_for(item["id"], connectors, output_port),
        }
        for item in core_items
    ]
    return {"schemaVersion": 1, "nodes": nodes}, target_item_id
