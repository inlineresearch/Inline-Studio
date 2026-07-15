"""Frames + takes + frame-inputs, ported from the Studio ``electron/main/frames/store.ts``.

A frame is the atomic unit: an optional input asset plus a history of generated takes; its hero take
is the Output that flows downstream. All frames live in a single auto-created default sequence.

Functions operate on an open ``sqlite3.Connection`` (the project.db Core owns). Fal creation
(``create_fal_frame`` / ``set_model`` / ``set_provider``) takes the model's ``kind`` + default
``params`` from the caller: the fal node definitions live studio-side (TypeScript ``src/shared/``)
and are never duplicated here.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

_FRAME_COLUMNS = (
    "id, sequence_id, name, kind, position, input_asset_id, hero_take_id, provider, model_id, "
    "params, workflow_template_id, comfy_workflow_name, comfy_workflow_ready, created_at, "
    "updated_at"
)


def _now() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    return str(uuid.uuid4())


def _parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_to_frame(row: sqlite3.Row) -> dict[str, Any]:
    provider = row["provider"]
    provider = provider if provider in ("fal", "unset", "core") else "comfy"
    return {
        "id": row["id"],
        "sequenceId": row["sequence_id"],
        "name": row["name"],
        "kind": row["kind"],
        "position": row["position"],
        "inputAssetId": row["input_asset_id"],
        "heroTakeId": row["hero_take_id"],
        "provider": provider,
        "modelId": row["model_id"],
        "params": _parse_params(row["params"]),
        "workflowTemplateId": row["workflow_template_id"],
        "comfyWorkflowName": row["comfy_workflow_name"],
        "comfyWorkflowReady": row["comfy_workflow_ready"] == 1,
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _row_to_take(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "frameId": row["frame_id"],
        "filePath": row["file_path"],
        "kind": row["kind"],
        "params": _parse_params(row["params"]),
        "comfyPromptId": row["comfy_prompt_id"],
        "createdAt": row["created_at"],
    }


def _row_to_input(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "frameId": row["frame_id"],
        "assetId": row["asset_id"],
        "sourceFrameId": row["source_frame_id"],
        "position": row["position"],
    }


def _project_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT id FROM project LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("No project is open.")
    return row["id"]


def default_sequence_id(conn: sqlite3.Connection) -> str:
    """The single default sequence frames are created in; created on first use."""
    row = conn.execute("SELECT id FROM sequences ORDER BY position LIMIT 1").fetchone()
    if row is not None:
        return row["id"]
    seq_id = _uuid()
    conn.execute(
        "INSERT INTO sequences (id, project_id, name, position) VALUES (?, ?, ?, 0)",
        (seq_id, _project_id(conn), "Main"),
    )
    return seq_id


def get_frame(conn: sqlite3.Connection, frame_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
    if row is None:
        raise ValueError("Frame not found.")
    return _row_to_frame(row)


def list_frames(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    seq_id = default_sequence_id(conn)
    rows = conn.execute(
        "SELECT * FROM frames WHERE sequence_id = ? ORDER BY position", (seq_id,)
    ).fetchall()
    return [_row_to_frame(r) for r in rows]


def _frame_count(conn: sqlite3.Connection, seq_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM frames WHERE sequence_id = ?", (seq_id,)
    ).fetchone()["n"]


def _insert_frame(conn: sqlite3.Connection, frame: dict[str, Any]) -> None:
    conn.execute(
        f"INSERT INTO frames ({_FRAME_COLUMNS}) VALUES "
        "(:id, :sequence_id, :name, :kind, :position, :input_asset_id, :hero_take_id, :provider, "
        ":model_id, :params, :workflow_template_id, :comfy_workflow_name, :comfy_workflow_ready, "
        ":created_at, :updated_at)",
        frame,
    )


def _new_frame_row(
    conn: sqlite3.Connection,
    *,
    kind: str,
    input_asset_id: str | None,
    provider: str,
    model_id: str | None = None,
    params: dict[str, Any] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    seq_id = default_sequence_id(conn)
    position = _frame_count(conn, seq_id)
    now = _now()
    return {
        "id": _uuid(),
        "sequence_id": seq_id,
        "name": name if name is not None else str(position + 1),
        "kind": kind,
        "position": position,
        "input_asset_id": input_asset_id,
        "hero_take_id": None,
        "provider": provider,
        "model_id": model_id,
        "params": json.dumps(params or {}),
        "workflow_template_id": None,
        "comfy_workflow_name": None,
        "comfy_workflow_ready": 0,
        "created_at": now,
        "updated_at": now,
    }


def add_from_asset(conn: sqlite3.Connection, asset_id: str) -> dict[str, Any]:
    asset = conn.execute("SELECT id, kind FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if asset is None:
        raise ValueError("Asset not found.")
    frame = _new_frame_row(conn, kind=asset["kind"], input_asset_id=asset_id, provider="unset")
    _insert_frame(conn, frame)
    conn.execute(
        "INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES (?, ?, ?, 0)",
        (_uuid(), frame["id"], asset_id),
    )
    return get_frame(conn, frame["id"])


def create_empty_frame(conn: sqlite3.Connection) -> dict[str, Any]:
    frame = _new_frame_row(conn, kind="image", input_asset_id=None, provider="unset")
    _insert_frame(conn, frame)
    return get_frame(conn, frame["id"])


def create_fal_frame(
    conn: sqlite3.Connection, model_id: str, kind: str, params: dict[str, Any], title: str
) -> dict[str, Any]:
    """Create a fal generation frame. ``kind``/``params``/``title`` come from the node def."""
    frame = _new_frame_row(
        conn, kind=kind, input_asset_id=None, provider="fal", model_id=model_id, params=params,
        name=title,
    )
    _insert_frame(conn, frame)
    return get_frame(conn, frame["id"])


def set_model(
    conn: sqlite3.Connection, frame_id: str, model_id: str, kind: str, params: dict[str, Any]
) -> dict[str, Any]:
    get_frame(conn, frame_id)
    conn.execute(
        "UPDATE frames SET model_id = ?, params = ?, kind = ?, updated_at = ? WHERE id = ?",
        (model_id, json.dumps(params), kind, _now(), frame_id),
    )
    return get_frame(conn, frame_id)


def set_provider(
    conn: sqlite3.Connection,
    frame_id: str,
    provider: str,
    model_id: str | None = None,
    kind: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    get_frame(conn, frame_id)
    now = _now()
    if provider == "fal":
        if not model_id or kind is None:
            raise ValueError("A fal provider needs a model id + kind from the node def.")
        conn.execute(
            "UPDATE frames SET provider = 'fal', model_id = ?, params = ?, kind = ?, "
            "updated_at = ? WHERE id = ?",
            (model_id, json.dumps(params or {}), kind, now, frame_id),
        )
    else:
        conn.execute(
            "UPDATE frames SET provider = 'comfy', model_id = NULL, updated_at = ? WHERE id = ?",
            (now, frame_id),
        )
    return get_frame(conn, frame_id)


def unlink_workflow(conn: sqlite3.Connection, frame_id: str) -> dict[str, Any]:
    """Detach a frame's ComfyUI workflow link (desktop legacy; a reset on the web path)."""
    get_frame(conn, frame_id)
    conn.execute(
        "UPDATE frames SET comfy_workflow_name = NULL, comfy_workflow_ready = 0, "
        "updated_at = ? WHERE id = ?",
        (_now(), frame_id),
    )
    return get_frame(conn, frame_id)


def set_fal_params(
    conn: sqlite3.Connection, frame_id: str, params: dict[str, Any]
) -> dict[str, Any]:
    get_frame(conn, frame_id)
    conn.execute(
        "UPDATE frames SET params = ?, updated_at = ? WHERE id = ?",
        (json.dumps(params or {}), _now(), frame_id),
    )
    return get_frame(conn, frame_id)


def rename_frame(conn: sqlite3.Connection, frame_id: str, name: str) -> dict[str, Any]:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Frame name is required.")
    get_frame(conn, frame_id)
    conn.execute(
        "UPDATE frames SET name = ?, updated_at = ? WHERE id = ?", (trimmed, _now(), frame_id)
    )
    return get_frame(conn, frame_id)


def reorder_frames(conn: sqlite3.Connection, ordered_ids: list[str]) -> None:
    now = _now()
    conn.executemany(
        "UPDATE frames SET position = ?, updated_at = ? WHERE id = ?",
        [(i, now, fid) for i, fid in enumerate(ordered_ids)],
    )


def delete_frame(conn: sqlite3.Connection, frame_id: str) -> None:
    conn.execute("DELETE FROM takes WHERE frame_id = ?", (frame_id,))
    conn.execute("DELETE FROM frame_inputs WHERE frame_id = ?", (frame_id,))
    items = conn.execute(
        "SELECT id FROM moodboard_items WHERE frame_id = ? AND type = 'frame'", (frame_id,)
    ).fetchall()
    for item in items:
        conn.execute(
            "DELETE FROM moodboard_connectors WHERE from_item_id = ? OR to_item_id = ?",
            (item["id"], item["id"]),
        )
    conn.execute("DELETE FROM moodboard_items WHERE frame_id = ? AND type = 'frame'", (frame_id,))
    conn.execute("DELETE FROM frames WHERE id = ?", (frame_id,))


def clone_frame(conn: sqlite3.Connection, frame_id: str) -> dict[str, Any]:
    src = get_frame(conn, frame_id)
    clone = _new_frame_row(
        conn,
        kind=src["kind"],
        input_asset_id=src["inputAssetId"],
        provider=src["provider"],
        model_id=src["modelId"],
        params=src["params"],
        name=f"{src['name']} copy",
    )
    _insert_frame(conn, clone)
    inputs = conn.execute(
        "SELECT asset_id, source_frame_id, position FROM frame_inputs WHERE frame_id = ?",
        (frame_id,),
    ).fetchall()
    conn.executemany(
        "INSERT INTO frame_inputs (id, frame_id, asset_id, source_frame_id, position) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (_uuid(), clone["id"], i["asset_id"], i["source_frame_id"], i["position"])
            for i in inputs
        ],
    )
    return get_frame(conn, clone["id"])


# --- inputs -------------------------------------------------------------------------------------


def _input_rows(conn: sqlite3.Connection, frame_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM frame_inputs WHERE frame_id = ? ORDER BY position", (frame_id,)
    ).fetchall()


def list_inputs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM frame_inputs ORDER BY frame_id, position").fetchall()
    return [_row_to_input(r) for r in rows]


def resolve_frame_file(
    conn: sqlite3.Connection, frame_id: str, seen: set[str] | None = None
) -> dict[str, str] | None:
    """A frame's output file: its hero take, else newest take, else its first input (an asset, or an
    upstream frame's output followed up the flow chain). Returns {filePath, kind} or None."""
    seen = seen if seen is not None else set()
    if frame_id in seen:
        return None
    seen.add(frame_id)
    row = conn.execute("SELECT hero_take_id FROM frames WHERE id = ?", (frame_id,)).fetchone()
    take = None
    if row and row["hero_take_id"]:
        take = conn.execute(
            "SELECT file_path, kind FROM takes WHERE id = ?", (row["hero_take_id"],)
        ).fetchone()
    if take is None:
        take = conn.execute(
            "SELECT file_path, kind FROM takes WHERE frame_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (frame_id,),
        ).fetchone()
    if take is not None:
        return {"filePath": take["file_path"], "kind": take["kind"]}
    for inp in _input_rows(conn, frame_id):
        if inp["asset_id"]:
            asset = conn.execute(
                "SELECT file_path, kind FROM assets WHERE id = ?", (inp["asset_id"],)
            ).fetchone()
            if asset is not None:
                return {"filePath": asset["file_path"], "kind": asset["kind"]}
        elif inp["source_frame_id"]:
            up = resolve_frame_file(conn, inp["source_frame_id"], seen)
            if up is not None:
                return up
    return None


def frame_input_media(conn: sqlite3.Connection, frame_id: str) -> list[dict[str, str]]:
    """A frame's inputs resolved to {filePath, kind}, in order (skipping unresolvable ones)."""
    out: list[dict[str, str]] = []
    for inp in _input_rows(conn, frame_id):
        if inp["asset_id"]:
            asset = conn.execute(
                "SELECT file_path, kind FROM assets WHERE id = ?", (inp["asset_id"],)
            ).fetchone()
            if asset is not None:
                out.append({"filePath": asset["file_path"], "kind": asset["kind"]})
        elif inp["source_frame_id"]:
            up = resolve_frame_file(conn, inp["source_frame_id"])
            if up is not None:
                out.append(up)
    return out


def add_input(conn: sqlite3.Connection, frame_id: str, asset_id: str) -> dict[str, Any]:
    get_frame(conn, frame_id)
    existing = _input_rows(conn, frame_id)
    dup = next((r for r in existing if r["asset_id"] == asset_id), None)
    if dup is not None:
        return _row_to_input(dup)
    row = {
        "id": _uuid(),
        "frameId": frame_id,
        "assetId": asset_id,
        "sourceFrameId": None,
        "position": len(existing),
    }
    conn.execute(
        "INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES (?, ?, ?, ?)",
        (row["id"], frame_id, asset_id, row["position"]),
    )
    return row


def add_inputs(
    conn: sqlite3.Connection, frame_id: str, asset_ids: list[str]
) -> list[dict[str, Any]]:
    get_frame(conn, frame_id)
    existing = _input_rows(conn, frame_id)
    have = {r["asset_id"] for r in existing if r["asset_id"]}
    added: list[dict[str, Any]] = []
    pos = len(existing)
    for asset_id in asset_ids:
        if asset_id in have:
            continue
        have.add(asset_id)
        row = {
            "id": _uuid(),
            "frameId": frame_id,
            "assetId": asset_id,
            "sourceFrameId": None,
            "position": pos,
        }
        conn.execute(
            "INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES (?, ?, ?, ?)",
            (row["id"], frame_id, asset_id, pos),
        )
        added.append(row)
        pos += 1
    return added


def add_source_input(
    conn: sqlite3.Connection, frame_id: str, source_frame_id: str
) -> dict[str, Any]:
    get_frame(conn, frame_id)
    get_frame(conn, source_frame_id)
    if frame_id == source_frame_id:
        raise ValueError("A frame cannot use its own output as input.")
    existing = _input_rows(conn, frame_id)
    dup = next((r for r in existing if r["source_frame_id"] == source_frame_id), None)
    if dup is not None:
        return _row_to_input(dup)
    row = {
        "id": _uuid(),
        "frameId": frame_id,
        "assetId": None,
        "sourceFrameId": source_frame_id,
        "position": len(existing),
    }
    conn.execute(
        "INSERT INTO frame_inputs (id, frame_id, asset_id, source_frame_id, position) "
        "VALUES (?, ?, NULL, ?, ?)",
        (row["id"], frame_id, source_frame_id, row["position"]),
    )
    return row


def remove_input(conn: sqlite3.Connection, frame_id: str, asset_id: str) -> None:
    conn.execute(
        "DELETE FROM frame_inputs WHERE frame_id = ? AND asset_id = ?", (frame_id, asset_id)
    )


def remove_input_by_id(conn: sqlite3.Connection, frame_id: str, input_id: str) -> None:
    conn.execute("DELETE FROM frame_inputs WHERE frame_id = ? AND id = ?", (frame_id, input_id))


def reorder_inputs(conn: sqlite3.Connection, frame_id: str, ordered_asset_ids: list[str]) -> None:
    conn.executemany(
        "UPDATE frame_inputs SET position = ? WHERE frame_id = ? AND asset_id = ?",
        [(i, frame_id, aid) for i, aid in enumerate(ordered_asset_ids)],
    )


# --- takes --------------------------------------------------------------------------------------


def list_takes(conn: sqlite3.Connection, frame_id: str) -> list[dict[str, Any]]:
    # rowid DESC breaks created_at ties deterministically (newest-inserted first).
    rows = conn.execute(
        "SELECT * FROM takes WHERE frame_id = ? ORDER BY created_at DESC, rowid DESC", (frame_id,)
    ).fetchall()
    return [_row_to_take(r) for r in rows]


def list_all_takes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM takes ORDER BY frame_id, created_at DESC, rowid DESC"
    ).fetchall()
    return [_row_to_take(r) for r in rows]


def hero_takes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT t.* FROM takes t JOIN frames s ON s.hero_take_id = t.id"
    ).fetchall()
    return [_row_to_take(r) for r in rows]


def set_hero(conn: sqlite3.Connection, frame_id: str, take_id: str | None) -> dict[str, Any]:
    get_frame(conn, frame_id)
    conn.execute(
        "UPDATE frames SET hero_take_id = ?, updated_at = ? WHERE id = ?",
        (take_id, _now(), frame_id),
    )
    return get_frame(conn, frame_id)


def add_take(
    conn: sqlite3.Connection,
    frame_id: str,
    file_path: str,
    kind: str,
    params: dict[str, Any],
    comfy_prompt_id: str | None = None,
) -> dict[str, Any]:
    """Insert a generated take for a frame and make it the hero (Output)."""
    get_frame(conn, frame_id)
    take_id = _uuid()
    now = _now()
    conn.execute(
        "INSERT INTO takes (id, frame_id, file_path, kind, params, comfy_prompt_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (take_id, frame_id, file_path, kind, json.dumps(params), comfy_prompt_id, now),
    )
    set_hero(conn, frame_id, take_id)
    return {
        "id": take_id,
        "frameId": frame_id,
        "filePath": file_path,
        "kind": kind,
        "params": params,
        "comfyPromptId": comfy_prompt_id,
        "createdAt": now,
    }


def delete_take(conn: sqlite3.Connection, take_id: str) -> str | None:
    """Delete a take (clearing it as hero). Returns its file_path so the caller can unlink it."""
    take = conn.execute("SELECT file_path FROM takes WHERE id = ?", (take_id,)).fetchone()
    if take is None:
        return None
    conn.execute("UPDATE frames SET hero_take_id = NULL WHERE hero_take_id = ?", (take_id,))
    conn.execute("DELETE FROM takes WHERE id = ?", (take_id,))
    return take["file_path"]
