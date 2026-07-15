"""The Studio assets + folders domain port."""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.studio import assets as ax
from inline_core.studio.schema import apply_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'Proj', 0, 0)")
    return c


def test_folder_crud_and_reparent(conn) -> None:
    root = ax.create_folder(conn, "Root", None)
    child = ax.create_folder(conn, "Child", root["id"])
    assert {f["name"] for f in ax.list_folders(conn)} == {"Root", "Child"}
    with pytest.raises(ValueError):
        ax.create_folder(conn, "  ", None)
    ax.rename_folder(conn, child["id"], "Renamed")
    assert ax.get_folder(conn, child["id"])["name"] == "Renamed"
    # Deleting Root reparents Child to Root's parent (None).
    ax.delete_folder(conn, root["id"])
    assert ax.get_folder(conn, child["id"])["parentId"] is None


def test_import_copies_file_and_inserts_row(conn, tmp_path) -> None:
    project = tmp_path / "proj.inlinestudio"
    project.mkdir()
    src = tmp_path / "pic.png"
    src.write_bytes(b"\x89PNG data")
    asset = ax.import_file(conn, project, str(src), None)
    assert asset is not None and asset["kind"] == "image"
    assert asset["name"] == "pic.png"
    copied = project / asset["filePath"]
    assert copied.is_file() and copied.read_bytes() == b"\x89PNG data"
    assert [a["id"] for a in ax.list_assets(conn)] == [asset["id"]]


def test_import_unknown_kind_returns_none(conn, tmp_path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hi")
    assert ax.import_file(conn, tmp_path, str(src), None) is None


def test_delete_asset_blocked_when_used_by_frame(conn) -> None:
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('a1', 'p', 'a', 'assets/a.png', 'image', 0)"
    )
    conn.execute(
        "INSERT INTO frames (id, sequence_id, name, kind, position, created_at, updated_at) "
        "VALUES ('f1', 's', '1', 'image', 0, 0, 0)"
    )
    conn.execute(
        "INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES ('i1', 'f1', 'a1', 0)"
    )
    with pytest.raises(ValueError):
        ax.delete_asset(conn, "a1")
    # Remove the input, then deletion succeeds and returns the file to unlink.
    conn.execute("DELETE FROM frame_inputs WHERE id = 'i1'")
    paths = ax.delete_asset(conn, "a1")
    assert paths == ["assets/a.png"]
    assert ax.list_assets(conn) == []
