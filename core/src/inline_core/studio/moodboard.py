"""Moodboard persistence — items (assets, text, frames, core nodes, layers, previews, director,
trim, prompt) and connectors — ported from the Studio ``electron/main/moodboard/store.ts``.

Operates on an open ``sqlite3.Connection``. The frame-creating adders compose with the frames
domain. Adders that need a fal node def (``add_gen_node``) take the model's kind/params/title from
the caller (the fal defs live studio-side).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from . import frames as fr

_ITEM_COLUMNS = (
    "id, project_id, type, asset_id, frame_id, parent_id, data, x, y, width, height, rotation, "
    "z_index, created_at, updated_at"
)

_DEFAULT_SIZE = {"image": (320, 180), "video": (360, 203), "audio": (320, 80)}

_DEFAULT_TEXT = {
    "text": "Text",
    "fontSize": 18,
    "bold": False,
    "italic": False,
    "underline": False,
    "color": "#e4e4e7",
    "align": "left",
}


def _now() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    return str(uuid.uuid4())


def _project_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT id FROM project LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("No project is open.")
    return row["id"]


def _parse(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "type": row["type"],
        "assetId": row["asset_id"],
        "frameId": row["frame_id"],
        "parentId": row["parent_id"],
        "data": _parse(row["data"]),
        "x": row["x"],
        "y": row["y"],
        "width": row["width"],
        "height": row["height"],
        "rotation": row["rotation"],
        "zIndex": row["z_index"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _row_to_connector(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "fromItemId": row["from_item_id"],
        "toItemId": row["to_item_id"],
        "label": row["label"],
        "data": _parse(row["data"]),
        "createdAt": row["created_at"],
    }


def get_item(conn: sqlite3.Connection, item_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM moodboard_items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        raise ValueError("Moodboard item not found.")
    return _row_to_item(row)


def list_items(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM moodboard_items").fetchall()
    return [_row_to_item(r) for r in rows]


def list_board(conn: sqlite3.Connection) -> dict[str, Any]:
    items = [_row_to_item(r) for r in conn.execute("SELECT * FROM moodboard_items").fetchall()]
    connectors = [
        _row_to_connector(r) for r in conn.execute("SELECT * FROM moodboard_connectors").fetchall()
    ]
    return {"items": items, "connectors": connectors}


def _next_z(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(z_index) AS z FROM moodboard_items").fetchone()
    return (row["z"] or 0) + 1


def _insert_item(
    conn: sqlite3.Connection,
    *,
    item_type: str,
    x: float,
    y: float,
    width: float,
    height: float,
    data: dict[str, Any] | None = None,
    asset_id: str | None = None,
    frame_id: str | None = None,
    z_index: int | None = None,
) -> dict[str, Any]:
    now = _now()
    item = {
        "id": _uuid(),
        "project_id": _project_id(conn),
        "type": item_type,
        "asset_id": asset_id,
        "frame_id": frame_id,
        "parent_id": None,
        "data": json.dumps(data or {}),
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": 0,
        "z_index": z_index if z_index is not None else _next_z(conn),
        "created_at": now,
        "updated_at": now,
    }
    conn.execute(
        f"INSERT INTO moodboard_items ({_ITEM_COLUMNS}) VALUES "
        "(:id, :project_id, :type, :asset_id, :frame_id, :parent_id, :data, :x, :y, :width, "
        ":height, :rotation, :z_index, :created_at, :updated_at)",
        item,
    )
    return get_item(conn, item["id"])


def add_asset(conn: sqlite3.Connection, asset_id: str, x: float, y: float) -> dict[str, Any]:
    asset = conn.execute("SELECT id, kind FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if asset is None:
        raise ValueError("Asset not found.")
    w, h = _DEFAULT_SIZE.get(asset["kind"], (320, 180))
    return _insert_item(conn, item_type="asset", x=x, y=y, width=w, height=h, asset_id=asset_id)


def add_text(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    return _insert_item(
        conn, item_type="text", x=x, y=y, width=200, height=60, data={"text": dict(_DEFAULT_TEXT)}
    )


def add_core_node(conn: sqlite3.Connection, core_type: str, x: float, y: float) -> dict[str, Any]:
    return _insert_item(
        conn, item_type="core", x=x, y=y, width=200, height=120,
        data={"core": {"type": core_type, "params": {}}},
    )


def add_frame_item(conn: sqlite3.Connection, frame_id: str, x: float, y: float) -> dict[str, Any]:
    return _insert_item(conn, item_type="frame", x=x, y=y, width=220, height=200, frame_id=frame_id)


def add_frame_from_asset(
    conn: sqlite3.Connection, asset_id: str, x: float, y: float
) -> dict[str, Any]:
    frame = fr.add_from_asset(conn, asset_id)
    return add_frame_item(conn, frame["id"], x, y)


def add_empty_frame(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    frame = fr.create_empty_frame(conn)
    return add_frame_item(conn, frame["id"], x, y)


def add_gen_node(
    conn: sqlite3.Connection,
    model_id: str,
    x: float,
    y: float,
    *,
    kind: str,
    params: dict[str, Any],
    title: str,
) -> dict[str, Any]:
    frame = fr.create_fal_frame(conn, model_id, kind, params, title)
    return _insert_item(
        conn, item_type="frame", x=x, y=y, width=240, height=380, frame_id=frame["id"]
    )


def add_layer(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    return _insert_item(
        conn, item_type="layer", x=x, y=y, width=420, height=300, data={"name": "Layer"}, z_index=0
    )


def add_preview(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    return _insert_item(conn, item_type="preview", x=x, y=y, width=280, height=220)


def add_director(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    return _insert_item(
        conn, item_type="director", x=x, y=y, width=440, height=400,
        data={"name": "Director", "director": {"width": 1920, "height": 1080, "fps": 30}},
    )


def add_trim(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    return _insert_item(
        conn, item_type="trim", x=x, y=y, width=360, height=170,
        data={"trim": {"inPoint": 0, "outPoint": 0}},
    )


def add_prompt(conn: sqlite3.Connection, x: float, y: float) -> dict[str, Any]:
    return _insert_item(
        conn, item_type="prompt", x=x, y=y, width=240, height=120, data={"promptText": ""}
    )


def update_item(conn: sqlite3.Connection, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    get_item(conn, item_id)  # ensure exists
    sets: list[str] = []
    params: dict[str, Any] = {"id": item_id, "updated_at": _now()}
    for key, column in (
        ("x", "x"), ("y", "y"), ("width", "width"), ("height", "height"),
        ("rotation", "rotation"), ("zIndex", "z_index"),
    ):
        value = patch.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            sets.append(f"{column} = :{column}")
            params[column] = value
    if "data" in patch:
        sets.append("data = :data")
        params["data"] = json.dumps(patch["data"])
    if "parentId" in patch:
        sets.append("parent_id = :parent_id")
        params["parent_id"] = patch["parentId"]
    sets.append("updated_at = :updated_at")
    conn.execute(f"UPDATE moodboard_items SET {', '.join(sets)} WHERE id = :id", params)
    return get_item(conn, item_id)


def set_core_node_output(conn: sqlite3.Connection, item_id: str, output: dict[str, Any]) -> None:
    """Store the latest render a Core media node produced, for display on the node."""
    try:
        item = get_item(conn, item_id)
    except ValueError:
        return
    core = (item.get("data") or {}).get("core")
    if item["type"] != "core" or not core:
        return
    update_item(conn, item_id, {"data": {**item["data"], "core": {**core, "output": output}}})


def delete_item(conn: sqlite3.Connection, item_id: str) -> None:
    conn.execute(
        "DELETE FROM moodboard_connectors WHERE from_item_id = ? OR to_item_id = ?",
        (item_id, item_id),
    )
    conn.execute("DELETE FROM moodboard_items WHERE id = ?", (item_id,))


def replace_board(
    conn: sqlite3.Connection, items: list[dict[str, Any]], connectors: list[dict[str, Any]]
) -> None:
    """Replace the whole board (undo/redo restore), preserving ids, in one transaction."""
    pid = _project_id(conn)
    conn.execute("DELETE FROM moodboard_connectors WHERE project_id = ?", (pid,))
    conn.execute("DELETE FROM moodboard_items WHERE project_id = ?", (pid,))
    for it in items:
        conn.execute(
            f"INSERT INTO moodboard_items ({_ITEM_COLUMNS}) VALUES "
            "(:id, :project_id, :type, :asset_id, :frame_id, :parent_id, :data, :x, :y, :width, "
            ":height, :rotation, :z_index, :created_at, :updated_at)",
            {
                "id": it["id"],
                "project_id": pid,
                "type": it["type"],
                "asset_id": it.get("assetId"),
                "frame_id": it.get("frameId"),
                "parent_id": it.get("parentId"),
                "data": json.dumps(it.get("data") or {}),
                "x": it["x"],
                "y": it["y"],
                "width": it["width"],
                "height": it["height"],
                "rotation": it.get("rotation", 0),
                "z_index": it.get("zIndex", 0),
                "created_at": it.get("createdAt", _now()),
                "updated_at": it.get("updatedAt", _now()),
            },
        )
    for c in connectors:
        conn.execute(
            "INSERT INTO moodboard_connectors "
            "(id, project_id, from_item_id, to_item_id, label, data, created_at) "
            "VALUES (:id, :project_id, :from_item_id, :to_item_id, :label, :data, :created_at)",
            {
                "id": c["id"],
                "project_id": pid,
                "from_item_id": c["fromItemId"],
                "to_item_id": c["toItemId"],
                "label": c.get("label"),
                "data": json.dumps(c.get("data") or {}),
                "created_at": c.get("createdAt", _now()),
            },
        )


def create_connector(
    conn: sqlite3.Connection,
    from_item_id: str,
    to_item_id: str,
    source_handle: str | None = None,
    target_handle: str | None = None,
) -> dict[str, Any]:
    get_item(conn, from_item_id)
    get_item(conn, to_item_id)
    connector = {
        "id": _uuid(),
        "projectId": _project_id(conn),
        "fromItemId": from_item_id,
        "toItemId": to_item_id,
        "label": None,
        "data": {"sourceHandle": source_handle, "targetHandle": target_handle},
        "createdAt": _now(),
    }
    conn.execute(
        "INSERT INTO moodboard_connectors "
        "(id, project_id, from_item_id, to_item_id, label, data, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            connector["id"], connector["projectId"], from_item_id, to_item_id, None,
            json.dumps(connector["data"]), connector["createdAt"],
        ),
    )
    return connector


def delete_connector(conn: sqlite3.Connection, connector_id: str) -> None:
    conn.execute("DELETE FROM moodboard_connectors WHERE id = ?", (connector_id,))


def set_connector_volume(conn: sqlite3.Connection, connector_id: str, volume: float) -> None:
    row = conn.execute(
        "SELECT data FROM moodboard_connectors WHERE id = ?", (connector_id,)
    ).fetchone()
    data = _parse(row["data"]) if row else {}
    data["volume"] = min(1.0, max(0.0, volume))
    conn.execute(
        "UPDATE moodboard_connectors SET data = ? WHERE id = ?",
        (json.dumps(data), connector_id),
    )


def prompt_text_for_frame(conn: sqlite3.Connection, frame_id: str) -> str | None:
    """The text of a Prompt node wired into this frame's canvas node on the 'prompt' handle."""
    nodes = conn.execute(
        "SELECT id FROM moodboard_items WHERE frame_id = ? AND type = 'frame'", (frame_id,)
    ).fetchall()
    for node in nodes:
        conns = conn.execute(
            "SELECT from_item_id, data FROM moodboard_connectors WHERE to_item_id = ?",
            (node["id"],),
        ).fetchall()
        for c in conns:
            if _parse(c["data"]).get("targetHandle") != "prompt":
                continue
            src = conn.execute(
                "SELECT data FROM moodboard_items WHERE id = ? AND type = 'prompt'",
                (c["from_item_id"],),
            ).fetchone()
            if src is None:
                continue
            text = _parse(src["data"]).get("promptText")
            if isinstance(text, str) and text.strip():
                return text
    return None
