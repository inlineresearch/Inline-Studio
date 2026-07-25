"""CRUD for the LoRA-training tables (``training_datasets`` / ``_dataset_items`` / ``_runs``).

Pure DB functions over an open ``sqlite3.Connection``, mirroring ``assets.py``. Row mappers emit the
camelCase shapes the wire contract (``TrainingDataset`` / ``TrainingDatasetItem`` / ``TrainingRun``
in ``src/shared/types.ts``) expects. The ``Training`` orchestrator (``training.py``) drives the
actual run; this module only owns the rows.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

# The hyperparam fields that also get their own columns for querying/display.
_DEFAULT_HYPERPARAMS: dict[str, Any] = {
    # Defaulted so a run row written before Krea 2 existed still resumes as Z-Image.
    "arch": "z-image",
    "baseMode": "turbo_adapter",
    "baseQuant": "auto",
    "loraScope": "full",
    "captionDropout": 0.05,
    "flipAugment": False,
    "rank": 16,
    "alpha": 16,
    "learningRate": 1e-4,
    "steps": 1500,
    "batchSize": 1,
    "resolution": 1024,
    "saveEvery": 250,
    "gpuIds": [],
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


# --- datasets -----------------------------------------------------------------------------------


def _row_to_dataset(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "triggerWord": row["trigger_word"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_datasets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM training_datasets ORDER BY created_at DESC").fetchall()
    return [_row_to_dataset(r) for r in rows]


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM training_datasets WHERE id = ?", (dataset_id,)).fetchone()
    if row is None:
        raise ValueError("Training dataset not found.")
    return _row_to_dataset(row)


def create_dataset(conn: sqlite3.Connection, name: str, trigger_word: str) -> dict[str, Any]:
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Dataset name is required.")
    now = _now()
    row = {
        "id": _uuid(),
        "project_id": _project_id(conn),
        "name": trimmed,
        "trigger_word": trigger_word.strip(),
        "created_at": now,
        "updated_at": now,
    }
    conn.execute(
        "INSERT INTO training_datasets "
        "(id, project_id, name, trigger_word, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        tuple(row.values()),
    )
    return get_dataset(conn, row["id"])


def _touch_dataset(conn: sqlite3.Connection, dataset_id: str) -> None:
    conn.execute(
        "UPDATE training_datasets SET updated_at = ? WHERE id = ?", (_now(), dataset_id)
    )


# --- dataset items ------------------------------------------------------------------------------


def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "datasetId": row["dataset_id"],
        "assetId": row["asset_id"],
        "caption": row["caption"],
        "position": row["position"],
        "createdAt": row["created_at"],
    }


def list_items(conn: sqlite3.Connection, dataset_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM training_dataset_items WHERE dataset_id = ? ORDER BY position",
        (dataset_id,),
    ).fetchall()
    return [_row_to_item(r) for r in rows]


def add_items(
    conn: sqlite3.Connection, dataset_id: str, asset_ids: list[str]
) -> list[dict[str, Any]]:
    """Append library assets as items, skipping any already in the dataset. Returns the new rows."""
    get_dataset(conn, dataset_id)  # validate
    existing = {
        r["asset_id"]
        for r in conn.execute(
            "SELECT asset_id FROM training_dataset_items WHERE dataset_id = ?", (dataset_id,)
        ).fetchall()
    }
    start = conn.execute(
        "SELECT COALESCE(MAX(position) + 1, 0) AS n FROM training_dataset_items "
        "WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()["n"]
    created: list[dict[str, Any]] = []
    position = start
    for asset_id in asset_ids:
        if asset_id in existing:
            continue
        existing.add(asset_id)
        item_id = _uuid()
        conn.execute(
            "INSERT INTO training_dataset_items (id, dataset_id, asset_id, caption, position, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, dataset_id, asset_id, "", position, _now()),
        )
        created.append(get_item(conn, item_id))
        position += 1
    if created:
        _touch_dataset(conn, dataset_id)
    return created


def get_item(conn: sqlite3.Connection, item_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM training_dataset_items WHERE id = ?", (item_id,)
    ).fetchone()
    if row is None:
        raise ValueError("Dataset item not found.")
    return _row_to_item(row)


def set_caption(conn: sqlite3.Connection, item_id: str, caption: str) -> dict[str, Any]:
    item = get_item(conn, item_id)
    conn.execute(
        "UPDATE training_dataset_items SET caption = ? WHERE id = ?", (caption, item_id)
    )
    _touch_dataset(conn, item["datasetId"])
    return get_item(conn, item_id)


def remove_item(conn: sqlite3.Connection, item_id: str) -> None:
    item = get_item(conn, item_id)
    conn.execute("DELETE FROM training_dataset_items WHERE id = ?", (item_id,))
    _touch_dataset(conn, item["datasetId"])


# --- runs ---------------------------------------------------------------------------------------


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    hyperparams = json.loads(row["hyperparams"] or "{}")
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "datasetId": row["dataset_id"],
        "name": row["name"],
        "status": row["status"],
        "hyperparams": {**_DEFAULT_HYPERPARAMS, **hyperparams},
        "outputLoraPath": row["output_lora_path"],
        "progressFraction": row["progress_fraction"],
        "progressStatus": row["progress_status"],
        "step": row["step"],
        "totalSteps": row["total_steps"],
        "error": row["error"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM training_runs ORDER BY created_at DESC").fetchall()
    return [_row_to_run(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError("Training run not found.")
    return _row_to_run(row)


def create_run(
    conn: sqlite3.Connection, dataset_id: str, name: str, hyperparams: dict[str, Any]
) -> dict[str, Any]:
    dataset = get_dataset(conn, dataset_id)
    merged = {**_DEFAULT_HYPERPARAMS, **hyperparams}
    now = _now()
    run_id = _uuid()
    conn.execute(
        "INSERT INTO training_runs (id, project_id, dataset_id, name, status, base_mode, "
        "hyperparams, total_steps, gpu_ids, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            dataset["projectId"],
            dataset_id,
            name.strip() or dataset["name"],
            str(merged["baseMode"]),
            json.dumps(merged),
            int(merged["steps"]),
            json.dumps(merged["gpuIds"]),
            now,
            now,
        ),
    )
    return get_run(conn, run_id)


# Columns the orchestrator may patch as a run progresses (camelCase key -> column).
_RUN_PATCH_COLUMNS = {
    "status": "status",
    "outputLoraPath": "output_lora_path",
    "progressFraction": "progress_fraction",
    "progressStatus": "progress_status",
    "step": "step",
    "totalSteps": "total_steps",
    "checkpointPath": "checkpoint_path",
    "error": "error",
}


def update_run(conn: sqlite3.Connection, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    sets: list[str] = []
    values: list[Any] = []
    for key, value in patch.items():
        column = _RUN_PATCH_COLUMNS.get(key)
        if column is None:
            continue
        sets.append(f"{column} = ?")
        values.append(value)
    sets.append("updated_at = ?")
    values.append(_now())
    values.append(run_id)
    conn.execute(f"UPDATE training_runs SET {', '.join(sets)} WHERE id = ?", tuple(values))
    return get_run(conn, run_id)
