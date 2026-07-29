"""Serialize a canvas subgraph into an Inline Core graph (schemaVersion 1) - ported from the Studio
``electron/main/core/workflow.ts``. Walks the connector graph upstream from a target node's closure:

- a ``core`` item   -> its Core node type + params (handles are already Core port ids)
- a ``prompt`` item -> an ``input/text`` source node
- an ``asset`` item -> an ``input/image`` source node (local path ref)
- a ``frame`` item  -> an ``input/image`` source node pointing at the frame's resolved output file
  (its hero take), so wiring a rendered frame into a Core node feeds that image without recomputing
  the frame. This is the closure boundary: upstream frames are frozen curated inputs, not re-run.

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


def _source_output_port(source: dict[str, Any] | None, source_handle: str | None) -> str:
    if source and source["type"] == "prompt":
        return "text"
    # An asset, a rendered frame, a Load Assets loader, or a Control Space render all become an
    # ``input/image`` source node.
    if source and source["type"] in ("asset", "frame", "loader", "controlSpace"):
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
    resolve_frame_path: Callable[[str], str | None],
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
    # A Load Assets loader feeds its hero (first) asset as a frozen image source.
    if item["type"] == "loader":
        asset_ids = data.get("assetIds") or []
        path = resolve_asset_path(asset_ids[0]) if asset_ids else None
        if not path:
            return None
        return {
            "id": item["id"],
            "type": "input/image",
            "params": {"asset": {"ref": "path", "path": path}},
        }
    # A Control Space node feeds its rendered OpenPose control map (a library asset) as a frozen
    # image source, wired into a gen node's control input.
    if item["type"] == "controlSpace":
        asset_id = data.get("controlAssetId")
        path = resolve_asset_path(asset_id) if asset_id else None
        if not path:
            return None
        return {
            "id": item["id"],
            "type": "input/image",
            "params": {"asset": {"ref": "path", "path": path}},
        }
    # A rendered frame wired into a Core node feeds its output image as a frozen source (its hero
    # take), so nothing upstream of the frame is recomputed.
    if item["type"] == "frame" and item.get("frameId"):
        path = resolve_frame_path(item["frameId"])
        if not path:
            return None
        return {
            "id": item["id"],
            "type": "input/image",
            "params": {"asset": {"ref": "path", "path": path}},
        }
    return None


def _facing_hint(item: dict[str, Any]) -> dict[str, str] | None:
    """The facing prompt text a Control Space node carries, if it is enabled and non-empty."""
    scene = (item.get("data") or {}).get("controlScene") or {}
    if scene.get("applyPromptHint") is False:
        return None
    hint = scene.get("promptHint")
    if not isinstance(hint, dict):
        return None
    positive = str(hint.get("positive") or "").strip()
    negative = str(hint.get("negative") or "").strip()
    return {"positive": positive, "negative": negative} if positive or negative else None


def _apply_facing_hints(nodes: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> None:
    """Fold a wired Control Space node's facing into the gen node's prompt and negative prompt.

    A control map has no channel that can say "this character faces away" - the pose ControlNet only
    sees a face-less skeleton, which is ambiguous, so the model falls back on its prior and renders
    the head turned back. The text encoder is the only place facing can be stated, so it is stated
    there. The prompt is rewired to a derived text node rather than editing the shared
    ``input/text`` in place, so a second gen node reading the same prompt node is unaffected.
    """
    by_node = {n["id"]: n for n in nodes}
    for node in list(nodes):
        item = by_id.get(node["id"])
        if item is None or item.get("type") != "core":
            continue
        inputs: dict[str, dict[str, str]] = node.get("inputs") or {}
        hint: dict[str, str] | None = None
        for edge in inputs.values():
            source_item = by_id.get(edge["from"])
            # Only when the control map itself made it into the graph - an unresolvable map means no
            # ControlNet runs, and a lone "from behind" in the prompt would be a lie.
            if edge["from"] not in by_node:
                continue
            if source_item is not None and source_item.get("type") == "controlSpace":
                hint = _facing_hint(source_item)
                if hint is not None:
                    break
        if hint is None:
            continue
        # Idempotent: a take's recipe records the params the run actually used, so restoring a take
        # writes an injected hint back onto the node. Appending again would stack it every render.
        if hint["negative"]:
            params = node.setdefault("params", {})
            existing = str(params.get("negative_prompt") or "").strip()
            if hint["negative"] not in existing:
                params["negative_prompt"] = (
                    f"{existing}, {hint['negative']}" if existing else hint["negative"]
                )
        prompt_edge = inputs.get("prompt")
        source = by_node.get(prompt_edge["from"]) if prompt_edge else None
        # Only a plain text source can be extended; a node-produced prompt is left alone.
        if not hint["positive"] or source is None or source["type"] != "input/text":
            continue
        text = str((source.get("params") or {}).get("text") or "").strip()
        if hint["positive"] in text:
            continue
        derived_id = f"{node['id']}::facing"
        nodes.append(
            {
                "id": derived_id,
                "type": "input/text",
                "params": {"text": f"{text}, {hint['positive']}" if text else hint["positive"]},
            }
        )
        inputs["prompt"] = {"from": derived_id, "output": "text"}


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

    def resolve_frame_path(frame_id: str) -> str | None:
        out = fr.resolve_frame_file(conn, frame_id)
        return str(folder / out["filePath"]) if out else None

    nodes: list[dict[str, Any]] = []
    for node_id in _upstream_closure(target_item_id, connectors):
        item = by_id.get(node_id)
        if item is None:
            continue
        node = _item_to_node(item, connectors, by_id, resolve_asset_path, resolve_frame_path)
        if node is not None:
            nodes.append(node)
    _apply_facing_hints(nodes, by_id)
    return {"schemaVersion": 1, "nodes": nodes}, target_item_id
