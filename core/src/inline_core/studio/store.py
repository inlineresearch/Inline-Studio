"""Project lifecycle + app-global settings/recents, ported from the Studio TS backend
(``electron/main/project/*`` + ``settings/store.ts``). Core owns ``project.db`` once this layer is
wired into ``/rpc``.

A project on disk is a portable folder::

    MyFilm.inlinestudio/
      project.db   assets/   takes/   thumbs/

The web SPA has no native folder picker, so new projects are created under a workspace dir (like the
legacy Node web server did with STORYLINE_WORKSPACE_DIR). Recents + settings are app-global JSON in
the app data dir.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .schema import apply_schema

PROJECT_EXT = ".inlinestudio"
LEGACY_EXTS = (".storyline",)
PROJECT_EXTS = (PROJECT_EXT, *LEGACY_EXTS)
SUBDIRS = ("assets", "takes", "thumbs")
MAX_RECENTS = 12


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sanitize_folder_name(name: str) -> str:
    base = re.sub(r"^-+|-+$", "", re.sub(r"[^\w.-]+", "-", name.strip()))
    return base or "untitled"


def _is_project_ext(folder: str) -> bool:
    return any(folder.endswith(ext) for ext in PROJECT_EXTS)


class StudioStore:
    """Owns the single open project (folder + SQLite connection) and app-global settings/recents."""

    def __init__(
        self,
        app_data_dir: str | Path,
        workspace_dir: str | Path,
        *,
        default_comfy_url: str = "http://127.0.0.1:8188",
        default_core_url: str = "http://127.0.0.1:8848",
    ) -> None:
        self._app_data = Path(app_data_dir)
        self._workspace = Path(workspace_dir)
        self._app_data.mkdir(parents=True, exist_ok=True)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._default_comfy_url = default_comfy_url
        self._default_core_url = default_core_url
        self._conn: sqlite3.Connection | None = None
        self._folder: Path | None = None
        self._current: dict[str, Any] | None = None

    # --- project db connection ------------------------------------------------------------------

    def _open_db(self, folder: Path) -> sqlite3.Connection:
        """Open (creating if needed) the project.db inside a project folder, applying the schema."""
        self.close()
        db_path = folder / "project.db"
        # A stale -shm in a copied project is rebuildable; drop it so SQLite regenerates it on open.
        shm = Path(f"{db_path}-shm")
        if shm.exists():
            try:
                shm.unlink()
            except OSError:
                pass
        # Autocommit (isolation_level=None) so each write persists immediately, matching the Node
        # backend's better-sqlite3 default — the domain modules don't manage transactions.
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        self._conn = conn
        self._folder = folder
        return conn

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("No project is open.")
        return self._conn

    # Back-compat internal alias.
    _db = conn

    def folder(self) -> Path:
        if self._folder is None:
            raise RuntimeError("No project is open.")
        return self._folder

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._folder = None

    # --- projects -------------------------------------------------------------------------------

    def create_project(self, name: str, parent_dir: str | None = None) -> dict[str, Any]:
        if not name or not name.strip():
            raise ValueError("Project name is required.")
        parent = Path(parent_dir) if parent_dir else self._workspace
        folder = parent / f"{_sanitize_folder_name(name)}{PROJECT_EXT}"
        if folder.exists():
            raise ValueError(f"A project already exists at {folder}")
        folder.mkdir(parents=True)
        for sub in SUBDIRS:
            (folder / sub).mkdir(parents=True, exist_ok=True)

        conn = self._open_db(folder)
        now = _now_ms()
        pid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO project (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (pid, name, now, now),
        )
        conn.commit()
        # TODO(B2/B3): seed the starter graph (empty frame + Prompt -> Z-Image) once the frames and
        # moodboard stores are ported. New projects open with an empty board until then.
        project = {"id": pid, "name": name, "path": str(folder), "createdAt": now, "updatedAt": now}
        self._current = project
        self.record_recent(name, str(folder))
        return project

    def open_project(self, selected: str) -> dict[str, Any]:
        folder = self.resolve_project_folder(selected)
        if folder is None:
            raise ValueError("That folder is not an Inline Studio project (no project.db found).")
        self._open_db(folder)
        for sub in SUBDIRS:
            (folder / sub).mkdir(parents=True, exist_ok=True)
        project = self._load_project_row(folder)
        self._current = project
        self.record_recent(project["name"], str(folder))
        return project

    def _load_project_row(self, folder: Path) -> dict[str, Any]:
        row = self._db().execute(
            "SELECT id, name, created_at, updated_at FROM project LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("project.db is missing its project record.")
        return {
            "id": row["id"],
            "name": row["name"],
            "path": str(folder),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def current_project(self) -> dict[str, Any] | None:
        return self._current

    def media_dirs(self) -> dict[str, str]:
        if self._folder is None:
            raise RuntimeError("No project is open.")
        return {
            "inputDir": str(self._folder / "assets"),
            "outputDir": str(self._folder / "takes"),
        }

    def resolve_project_folder(self, folder: str) -> Path | None:
        """The real project folder for a picked one: itself if it holds project.db, else a child
        that does (an unzip may wrap the project one level down). Prefers a *.inlinestudio child."""
        root = Path(folder)
        try:
            if (root / "project.db").exists():
                return root
            children = [c for c in root.iterdir() if c.is_dir()]
            preferred = next(
                (c for c in children if _is_project_ext(c.name) and (c / "project.db").exists()),
                None,
            )
            if preferred is not None:
                return preferred
            any_child = next((c for c in children if (c / "project.db").exists()), None)
            return any_child
        except OSError:
            return None

    def is_project_folder(self, folder: str) -> bool:
        return self.resolve_project_folder(folder) is not None

    # --- recents (app-global JSON) --------------------------------------------------------------

    def _recents_file(self) -> Path:
        return self._app_data / "recent-projects.json"

    def list_recent(self) -> list[dict[str, Any]]:
        file = self._recents_file()
        if not file.exists():
            return []
        try:
            parsed = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return parsed if isinstance(parsed, list) else []

    def record_recent(self, name: str, path: str) -> None:
        now = _now_ms()
        existing = [r for r in self.list_recent() if r.get("path") != path]
        nxt = [{"name": name, "path": path, "lastOpenedAt": now}, *existing][:MAX_RECENTS]
        self._recents_file().write_text(json.dumps(nxt, indent=2), encoding="utf-8")

    # --- settings (app-global JSON) -------------------------------------------------------------

    def _settings_file(self) -> Path:
        return self._app_data / "settings.json"

    def _read_settings(self) -> dict[str, Any]:
        file = self._settings_file()
        if not file.exists():
            return {}
        try:
            parsed = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _non_empty(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    def get_settings(self) -> dict[str, str]:
        saved = self._read_settings()
        return {
            "comfyUrl": self._non_empty(saved.get("comfyUrl")) or self._default_comfy_url,
            "coreUrl": self._non_empty(saved.get("coreUrl")) or self._default_core_url,
        }

    def _save_settings(self, settings: dict[str, str]) -> dict[str, str]:
        self._settings_file().write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return settings

    def set_comfy_url(self, url: str) -> dict[str, str]:
        settings = self.get_settings()
        settings["comfyUrl"] = url.strip() or self._default_comfy_url
        return self._save_settings(settings)

    def set_core_url(self, url: str) -> dict[str, str]:
        settings = self.get_settings()
        settings["coreUrl"] = url.strip() or self._default_core_url
        return self._save_settings(settings)

    # --- fal API key (app-global, server-side only) ---------------------------------------------

    def _fal_key_file(self) -> Path:
        return self._app_data / "fal_key"

    def fal_key(self) -> str | None:
        file = self._fal_key_file()
        if not file.exists():
            env = os.environ.get("FAL_KEY", "").strip()
            return env or None
        try:
            return file.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def fal_status(self) -> dict[str, bool]:
        # `encrypted` is False: the key lives in a 0600 file, not OS-encrypted (single-user local).
        return {"hasKey": self.fal_key() is not None, "encrypted": False}

    def set_fal_key(self, key: str) -> dict[str, bool]:
        file = self._fal_key_file()
        file.write_text(key.strip(), encoding="utf-8")
        try:
            file.chmod(0o600)  # owner-only: the key never leaves the server side
        except OSError:
            pass
        return self.fal_status()

    def clear_fal_key(self) -> dict[str, bool]:
        try:
            self._fal_key_file().unlink(missing_ok=True)
        except OSError:
            pass
        return {"hasKey": False, "encrypted": False}
