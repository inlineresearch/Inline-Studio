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
    # The v9 rebuild copies a fixed column list, so `handle` must be added after it, not dropped.
    assert "handle" in _columns(conn, "frame_inputs")


def test_frame_inputs_gain_a_nullable_handle_keeping_existing_rows() -> None:
    """v16 -> v17: existing inputs survive and read back untagged, so they still resolve by kind."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE frame_inputs (id TEXT PRIMARY KEY, frame_id TEXT NOT NULL, asset_id TEXT,
                                   source_frame_id TEXT, position INTEGER NOT NULL);
        INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES ('i1', 'f1', 'a1', 0);
        PRAGMA user_version = 16;
        """
    )
    apply_schema(conn)
    assert "handle" in _columns(conn, "frame_inputs")
    row = conn.execute("SELECT asset_id, handle FROM frame_inputs WHERE id='i1'").fetchone()
    assert row == ("a1", None)


def test_generation_runs_table_is_added_to_an_existing_project() -> None:
    """v17 -> v18: run history is per project, so it arrives on upgrade, not only on create."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE project (id TEXT PRIMARY KEY, name TEXT, created_at INTEGER,
                              updated_at INTEGER);
        INSERT INTO project (id, name, created_at, updated_at) VALUES ('p1', 'Alpha', 0, 0);
        PRAGMA user_version = 17;
        """
    )

    apply_schema(conn)

    assert "generation_runs" in _tables(conn)
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    # The existing row survived the upgrade.
    assert conn.execute("SELECT name FROM project").fetchone()[0] == "Alpha"
    conn.execute(
        "INSERT INTO generation_runs (id, project_id, item_id, status, queued_at) "
        "VALUES ('r1', 'p1', 'i1', 'done', 1)"
    )
    assert conn.execute("SELECT status FROM generation_runs").fetchone()[0] == "done"


def test_v18_project_gains_the_motion_lora_columns(tmp_path) -> None:
    """A pre-19 project must open and read as a clip-mode dataset with no references, rather than
    needing a rebuild. Both columns are additive and defaulted for exactly that reason."""
    import sqlite3

    from inline_core.studio.schema import apply_schema

    db = tmp_path / "v18.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE training_datasets (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
          trigger_word TEXT NOT NULL DEFAULT '',
          created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
        CREATE TABLE training_dataset_items (
          id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, asset_id TEXT NOT NULL,
          caption TEXT NOT NULL DEFAULT '', position INTEGER NOT NULL,
          created_at INTEGER NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO training_datasets VALUES ('d1','p1','old','trigger',0,0)"
    )
    conn.execute(
        "INSERT INTO training_dataset_items VALUES ('i1','d1','a1','a caption',0,0)"
    )
    conn.commit()

    apply_schema(conn)

    dataset = conn.execute("SELECT mode FROM training_datasets WHERE id='d1'").fetchone()
    item = conn.execute(
        "SELECT reference_asset_id, caption FROM training_dataset_items WHERE id='i1'"
    ).fetchone()
    assert dataset[0] == "clip"
    assert item[0] is None
    assert item[1] == "a caption", "the existing row survived the migration"
    conn.close()


def test_v19_trainer_canvas_merges_clear_of_the_studio_graph(tmp_path) -> None:
    """The two canvases both start at the origin, so a merge in place would stack the training graph
    on the generation graph. Types are namespaced in the same pass."""
    import sqlite3

    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(tmp_path / "v19.db")
    conn.execute("PRAGMA user_version = 19")
    conn.executescript(
        """
        CREATE TABLE moodboard_items (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, surface TEXT NOT NULL DEFAULT 'studio',
          type TEXT NOT NULL DEFAULT 'asset', asset_id TEXT, data TEXT,
          x REAL NOT NULL, y REAL NOT NULL, width REAL NOT NULL, height REAL NOT NULL);
        CREATE TABLE moodboard_connectors (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, surface TEXT NOT NULL DEFAULT 'studio',
          from_item_id TEXT NOT NULL, to_item_id TEXT NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO moodboard_items (id, project_id, surface, type, x, y, width, height)"
        " VALUES ('g1','p1','studio','core',100,0,400,300)"
    )
    conn.execute(
        "INSERT INTO moodboard_items (id, project_id, surface, type, x, y, width, height)"
        " VALUES ('t1','p1','trainer','trainer',60,150,420,340)"
    )
    conn.execute(
        "INSERT INTO moodboard_connectors (id, project_id, surface, from_item_id, to_item_id)"
        " VALUES ('c1','p1','trainer','t1','t1')"
    )
    conn.commit()

    apply_schema(conn)

    moved = conn.execute("SELECT surface, type, x, y FROM moodboard_items WHERE id='t1'").fetchone()
    kept = conn.execute("SELECT surface, x FROM moodboard_items WHERE id='g1'").fetchone()
    edge = conn.execute("SELECT surface FROM moodboard_connectors WHERE id='c1'").fetchone()
    assert moved[0] == "studio"
    assert moved[1] == "train/lora", "the type is namespaced, not left generic"
    assert moved[2] >= kept[1] + 400, "the training graph lands right of the generation graph"
    assert moved[3] == 150, "y is untouched"
    assert kept[0] == "studio" and kept[1] == 100, "studio rows do not move"
    assert edge[0] == "studio"
    conn.close()


def test_v20_the_loss_curve_edge_follows_its_port_rename(tmp_path) -> None:
    """React Flow anchors an edge by handle id, so a wire naming a port the node no longer has
    stops drawing altogether rather than drawing in the wrong place."""
    import json
    import sqlite3

    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(tmp_path / "v20.db")
    conn.execute("PRAGMA user_version = 20")
    conn.executescript(
        """
        CREATE TABLE moodboard_connectors (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, surface TEXT NOT NULL DEFAULT 'studio',
          from_item_id TEXT NOT NULL, to_item_id TEXT NOT NULL, label TEXT, data TEXT);
        """
    )
    for cid, data in (
        ("c1", {"sourceHandle": "run", "targetHandle": "run"}),
        ("c2", {"sourceHandle": "dataset", "targetHandle": "dataset"}),
        ("c3", None),
    ):
        conn.execute(
            "INSERT INTO moodboard_connectors (id, project_id, from_item_id, to_item_id, data)"
            " VALUES (?,?,?,?,?)",
            (cid, "p1", "a", "b", json.dumps(data) if data else None),
        )
    conn.commit()

    apply_schema(conn)

    rows = dict(conn.execute("SELECT id, data FROM moodboard_connectors").fetchall())
    assert json.loads(rows["c1"]) == {"sourceHandle": "metrics", "targetHandle": "metrics"}
    assert json.loads(rows["c2"])["sourceHandle"] == "dataset", "other ports are untouched"
    assert rows["c3"] is None, "an edge with no handles survives"
    conn.close()

