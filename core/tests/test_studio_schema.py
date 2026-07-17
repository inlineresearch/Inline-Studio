"""The Studio project.db schema port: fresh-DB shape, version stamping, and legacy migrations."""

from __future__ import annotations

import sqlite3

from inline_core.studio.schema import SCHEMA_VERSION, apply_schema

_TABLES = {
    "project",
    "sequences",
    "frames",
    "takes",
    "frame_inputs",
    "asset_folders",
    "assets",
    "moodboard_items",
    "moodboard_connectors",
    "workflow_templates",
    "pending_generation",
}


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_fresh_db_has_all_tables_and_version() -> None:
    conn = sqlite3.connect(":memory:")
    apply_schema(conn)
    assert _TABLES.issubset(_tables(conn))
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION


def test_apply_is_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    apply_schema(conn)
    apply_schema(conn)  # second run must not raise or duplicate
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION


def test_shot_to_frame_rename_migration() -> None:
    conn = sqlite3.connect(":memory:")
    # An old (pre-v8) DB used "shots" and shot_id columns.
    conn.executescript(
        """
        CREATE TABLE shots (id TEXT PRIMARY KEY, sequence_id TEXT, name TEXT, kind TEXT,
                            position INTEGER, created_at INTEGER, updated_at INTEGER);
        CREATE TABLE takes (id TEXT PRIMARY KEY, shot_id TEXT, file_path TEXT, kind TEXT,
                            params TEXT, created_at INTEGER);
        CREATE TABLE moodboard_items (id TEXT PRIMARY KEY, project_id TEXT, type TEXT, shot_id TEXT,
                                      x REAL, y REAL, width REAL, height REAL);
        INSERT INTO moodboard_items (id, project_id, type, x, y, width, height)
          VALUES ('m1', 'p', 'shot', 0, 0, 1, 1);
        PRAGMA user_version = 7;
        """
    )
    apply_schema(conn)
    tbls = _tables(conn)
    assert "frames" in tbls and "shots" not in tbls
    assert "frame_id" in _columns(conn, "takes")
    assert conn.execute("SELECT type FROM moodboard_items WHERE id='m1'").fetchone()[0] == "frame"


def test_additive_column_migration() -> None:
    conn = sqlite3.connect(":memory:")
    # An old assets table without folder_id / preview_path.
    conn.executescript(
        """
        CREATE TABLE assets (id TEXT PRIMARY KEY, project_id TEXT, name TEXT, file_path TEXT,
                             kind TEXT, thumb_path TEXT, created_at INTEGER);
        PRAGMA user_version = 1;
        """
    )
    apply_schema(conn)
    cols = _columns(conn, "assets")
    assert {"folder_id", "preview_path"}.issubset(cols)


def test_frame_inputs_asset_id_relaxed_to_nullable() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE frame_inputs (id TEXT PRIMARY KEY, frame_id TEXT NOT NULL,
                                   asset_id TEXT NOT NULL, position INTEGER NOT NULL);
        PRAGMA user_version = 8;
        """
    )
    apply_schema(conn)
    cols = conn.execute("PRAGMA table_info(frame_inputs)").fetchall()
    asset = next(c for c in cols if c[1] == "asset_id")
    assert asset[3] == 0  # notnull flag cleared
