"""SQLite schema for a project's ``project.db`` — a faithful port of the Studio TypeScript
``electron/main/db/schema.ts`` (SCHEMA_VERSION 14). The DB is the source of truth for a project;
"save" is implicit. Bumping ``SCHEMA_VERSION`` + adding a migration is how the schema evolves.

Kept byte-compatible with the Node schema so Core can open existing ``.inlinestudio`` projects: same
tables, same column names, same ``user_version`` stamping, and the same additive/rename migrations.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 14

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  position    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
  id                   TEXT PRIMARY KEY,
  sequence_id          TEXT NOT NULL,
  name                 TEXT NOT NULL,
  kind                 TEXT NOT NULL,
  position             INTEGER NOT NULL,
  input_asset_id       TEXT,
  hero_take_id         TEXT,
  provider             TEXT NOT NULL DEFAULT 'comfy',
  model_id             TEXT,
  params               TEXT NOT NULL DEFAULT '{}',
  workflow_template_id TEXT,
  comfy_workflow_name  TEXT,
  comfy_workflow_ready INTEGER NOT NULL DEFAULT 0,
  created_at           INTEGER NOT NULL,
  updated_at           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS takes (
  id              TEXT PRIMARY KEY,
  frame_id         TEXT NOT NULL,
  file_path       TEXT NOT NULL,
  kind            TEXT NOT NULL,
  params          TEXT NOT NULL,
  comfy_prompt_id TEXT,
  created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frame_inputs (
  id             TEXT PRIMARY KEY,
  frame_id        TEXT NOT NULL,
  asset_id       TEXT,
  source_frame_id TEXT,
  position       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS asset_folders (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  parent_id   TEXT,
  created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL,
  folder_id    TEXT,
  name         TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  kind         TEXT NOT NULL,
  thumb_path   TEXT,
  preview_path TEXT,
  created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS moodboard_items (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  type        TEXT NOT NULL DEFAULT 'asset',
  asset_id    TEXT,
  frame_id     TEXT,
  parent_id   TEXT,
  data        TEXT,
  x           REAL NOT NULL,
  y           REAL NOT NULL,
  width       REAL NOT NULL,
  height      REAL NOT NULL,
  rotation    REAL NOT NULL DEFAULT 0,
  z_index     INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL DEFAULT 0,
  updated_at  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS moodboard_connectors (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL,
  from_item_id TEXT NOT NULL,
  to_item_id   TEXT NOT NULL,
  label        TEXT,
  data         TEXT,
  created_at   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS workflow_templates (
  id          TEXT PRIMARY KEY,
  project_id  TEXT NOT NULL,
  name        TEXT NOT NULL,
  graph       TEXT NOT NULL,
  params      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_generation (
  id           TEXT PRIMARY KEY,
  frame_id     TEXT NOT NULL,
  model_id     TEXT NOT NULL,
  endpoint     TEXT NOT NULL,
  request_id   TEXT NOT NULL,
  status_url   TEXT NOT NULL,
  response_url TEXT NOT NULL,
  params       TEXT NOT NULL,
  created_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_sequence ON frames(sequence_id);
CREATE INDEX IF NOT EXISTS idx_takes_frame ON takes(frame_id);
CREATE INDEX IF NOT EXISTS idx_frame_inputs_frame ON frame_inputs(frame_id);
CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id);
CREATE INDEX IF NOT EXISTS idx_assets_folder ON assets(folder_id);
CREATE INDEX IF NOT EXISTS idx_asset_folders_parent ON asset_folders(parent_id);
CREATE INDEX IF NOT EXISTS idx_moodboard_items_project ON moodboard_items(project_id);
CREATE INDEX IF NOT EXISTS idx_moodboard_connectors_project ON moodboard_connectors(project_id);
"""


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create tables (idempotent) and stamp the schema version. Mirrors ``applySchema``."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    from_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    # Additive column migrations run before SCHEMA_SQL (its indexes reference new columns).
    _migrate_columns(conn)
    conn.executescript(SCHEMA_SQL)
    _run_data_migrations(conn, from_version)
    _stamp_version(conn)
    conn.commit()


def _run_data_migrations(conn: sqlite3.Connection, from_version: int) -> None:
    # v5 -> v6: move each frame's single input_asset_id into the new frame_inputs table. Idempotent.
    if from_version < 6:
        conn.execute(
            """
            INSERT INTO frame_inputs (id, frame_id, asset_id, position)
            SELECT lower(hex(randomblob(16))), id, input_asset_id, 0
            FROM frames
            WHERE input_asset_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM frame_inputs si WHERE si.frame_id = frames.id);
            """
        )


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Additive column + rename migrations for pre-existing projects (no-ops on fresh DBs)."""
    _migrate_renames(conn)  # v7 -> v8: shot -> frame

    _add_column_if_missing(conn, "assets", "folder_id", "TEXT")

    _add_column_if_missing(conn, "moodboard_items", "type", "TEXT NOT NULL DEFAULT 'asset'")
    _add_column_if_missing(conn, "moodboard_items", "data", "TEXT")
    _add_column_if_missing(conn, "moodboard_items", "rotation", "REAL NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "moodboard_items", "z_index", "INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "moodboard_items", "created_at", "INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "moodboard_items", "updated_at", "INTEGER NOT NULL DEFAULT 0")

    _add_column_if_missing(conn, "frames", "input_asset_id", "TEXT")
    _add_column_if_missing(conn, "frames", "comfy_workflow_name", "TEXT")

    _add_column_if_missing(conn, "moodboard_items", "frame_id", "TEXT")
    _add_column_if_missing(conn, "moodboard_items", "parent_id", "TEXT")
    _add_column_if_missing(conn, "frame_inputs", "source_frame_id", "TEXT")

    _relax_frame_inputs_asset_id(conn)  # v8 -> v9: asset_id must be nullable

    _add_column_if_missing(conn, "assets", "preview_path", "TEXT")
    _add_column_if_missing(conn, "frames", "comfy_workflow_ready", "INTEGER NOT NULL DEFAULT 0")

    # v12 -> v13: fal provider + model id + params.
    _add_column_if_missing(conn, "frames", "provider", "TEXT NOT NULL DEFAULT 'comfy'")
    _add_column_if_missing(conn, "frames", "model_id", "TEXT")
    _add_column_if_missing(conn, "frames", "params", "TEXT NOT NULL DEFAULT '{}'")


def _relax_frame_inputs_asset_id(conn: sqlite3.Connection) -> None:
    """Rebuild frame_inputs to drop a legacy NOT NULL on asset_id. Idempotent."""
    if not _table_exists(conn, "frame_inputs"):
        return
    cols = conn.execute("PRAGMA table_info(frame_inputs)").fetchall()
    asset = next((c for c in cols if c[1] == "asset_id"), None)
    if asset is None or asset[3] == 0:  # c[3] = notnull flag
        return
    conn.executescript(
        """
        CREATE TABLE frame_inputs_new (
          id              TEXT PRIMARY KEY,
          frame_id        TEXT NOT NULL,
          asset_id        TEXT,
          source_frame_id TEXT,
          position        INTEGER NOT NULL
        );
        INSERT INTO frame_inputs_new (id, frame_id, asset_id, source_frame_id, position)
          SELECT id, frame_id, asset_id, source_frame_id, position FROM frame_inputs;
        DROP TABLE frame_inputs;
        ALTER TABLE frame_inputs_new RENAME TO frame_inputs;
        """
    )


def _migrate_renames(conn: sqlite3.Connection) -> None:
    """v7 -> v8: the "shot" domain was renamed to "frame". Guarded, so a no-op on fresh DBs."""
    if _table_exists(conn, "shots") and not _table_exists(conn, "frames"):
        conn.execute("ALTER TABLE shots RENAME TO frames")
    if _table_exists(conn, "shot_inputs") and not _table_exists(conn, "frame_inputs"):
        conn.execute("ALTER TABLE shot_inputs RENAME TO frame_inputs")
    _rename_column_if_exists(conn, "frame_inputs", "shot_id", "frame_id")
    _rename_column_if_exists(conn, "frame_inputs", "source_shot_id", "source_frame_id")
    _rename_column_if_exists(conn, "takes", "shot_id", "frame_id")
    _rename_column_if_exists(conn, "moodboard_items", "shot_id", "frame_id")
    if _table_exists(conn, "moodboard_items"):
        conn.execute("UPDATE moodboard_items SET type='frame' WHERE type='shot'")
    conn.executescript(
        "DROP INDEX IF EXISTS idx_shots_sequence;"
        "DROP INDEX IF EXISTS idx_takes_shot;"
        "DROP INDEX IF EXISTS idx_shot_inputs_shot;"
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _rename_column_if_exists(conn: sqlite3.Connection, table: str, old: str, new: str) -> None:
    if not _table_exists(conn, table):
        return
    cols = _column_names(conn, table)
    if old in cols and new not in cols:
        conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    if not _table_exists(conn, table):
        return
    if column not in _column_names(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _stamp_version(conn: sqlite3.Connection) -> None:
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
