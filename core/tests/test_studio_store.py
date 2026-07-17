"""The Studio project store: create/open projects, media dirs, recents, and settings."""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.studio.store import StudioStore


def _store(tmp_path):
    return StudioStore(tmp_path / "appdata", tmp_path / "workspace")


def test_create_project_makes_folder_db_and_subdirs(tmp_path) -> None:
    store = _store(tmp_path)
    project = store.create_project("My Film")

    folder = tmp_path / "workspace" / "My-Film.inlinestudio"
    assert (folder / "project.db").is_file()
    for sub in ("assets", "takes", "thumbs"):
        assert (folder / sub).is_dir()
    assert project["name"] == "My Film"
    assert project["path"] == str(folder)
    assert store.current_project()["id"] == project["id"]

    # The project row is queryable and matches.
    row = sqlite3.connect(str(folder / "project.db")).execute(
        "SELECT id, name FROM project"
    ).fetchone()
    assert row == (project["id"], "My Film")


def test_create_rejects_blank_name_and_duplicate(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create_project("   ")
    store.create_project("Dup")
    with pytest.raises(ValueError):
        store.create_project("Dup")


def test_open_project_loads_the_same_row(tmp_path) -> None:
    created = _store(tmp_path).create_project("Reopen Me")
    # A fresh store instance opens the folder from disk.
    reopened = _store(tmp_path).open_project(created["path"])
    assert reopened["id"] == created["id"]
    assert reopened["name"] == "Reopen Me"


def test_open_rejects_non_project_folder(tmp_path) -> None:
    (tmp_path / "not-a-project").mkdir()
    with pytest.raises(ValueError):
        _store(tmp_path).open_project(str(tmp_path / "not-a-project"))


def test_resolve_project_folder_unwraps_nested(tmp_path) -> None:
    store = _store(tmp_path)
    created = store.create_project("Nested")
    inner = created["path"]  # .../Nested.inlinestudio
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    # Simulate an unzip that wrapped the project one level down.
    import shutil

    shutil.move(inner, wrapper / "Nested.inlinestudio")
    resolved = store.resolve_project_folder(str(wrapper))
    assert resolved is not None and resolved.name == "Nested.inlinestudio"


def test_media_dirs_requires_open_project(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(RuntimeError):
        store.media_dirs()
    project = store.create_project("Media")
    dirs = store.media_dirs()
    assert dirs["inputDir"] == str(project["path"]) + "/assets"
    assert dirs["outputDir"] == str(project["path"]) + "/takes"


def test_recents_dedupe_and_cap(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(15):
        store.record_recent(f"P{i}", f"/path/{i}")
    recents = store.list_recent()
    assert len(recents) == 12  # MAX_RECENTS
    assert recents[0]["name"] == "P14"  # newest first
    # Re-recording an existing path moves it to the front without duplicating.
    store.record_recent("P5-again", "/path/5")
    recents = store.list_recent()
    assert recents[0]["path"] == "/path/5"
    assert sum(1 for r in recents if r["path"] == "/path/5") == 1


def test_settings_defaults_and_overrides(tmp_path) -> None:
    store = StudioStore(
        tmp_path / "a",
        tmp_path / "w",
        default_comfy_url="http://comfy",
        default_core_url="http://core",
    )
    assert store.get_settings() == {"comfyUrl": "http://comfy", "coreUrl": "http://core"}
    assert store.set_core_url("http://127.0.0.1:9999")["coreUrl"] == "http://127.0.0.1:9999"
    # Persisted + comfy still defaulted.
    assert store.get_settings()["coreUrl"] == "http://127.0.0.1:9999"
    assert store.get_settings()["comfyUrl"] == "http://comfy"
    # Blank falls back to the default.
    assert store.set_comfy_url("  ")["comfyUrl"] == "http://comfy"
