"""Orchestrate rendering a director timeline: preview (low-res proxy) + export (full-res), plus
the hero-take folder export. Ports ``electron/main/timeline/compose.ts`` + ``export/folder.ts``.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any

from .. import moodboard as mb
from .compose import ComposeSettings, build_compose_args, timeline_duration
from .ffmpeg import compose_render, ffmpeg_available
from .resolve import resolve_timeline, resolve_trim

_DEFAULT = {"width": 1920, "height": 1080, "fps": 30}


def _even(n: float) -> int:
    return max(2, round(n / 2) * 2)


def _clamp01(v: Any) -> float:
    return min(1.0, max(0.0, float(v))) if isinstance(v, (int, float)) else 1.0


class Timeline:
    def __init__(self, store: Any, events: Any) -> None:
        self._store = store
        self._events = events

    def resolve(self, owner_item_id: str) -> dict[str, Any]:
        display, _ = resolve_timeline(self._store.conn(), self._store.folder(), owner_item_id)
        return display

    def resolve_trim(self, item_id: str) -> dict[str, Any] | None:
        return resolve_trim(self._store.conn(), self._store.folder(), item_id)

    def set_volumes(self, owner_item_id: str, l1: Any, l2: Any) -> None:
        conn = self._store.conn()
        item = mb.get_item(conn, owner_item_id)
        mb.update_item(
            conn, owner_item_id,
            {"data": {**item["data"], "l1Volume": _clamp01(l1), "l2Volume": _clamp01(l2)}},
        )

    async def build_preview(self, owner_item_id: str) -> str | None:
        return await self._render(owner_item_id, preview=True)

    async def export(self, owner_item_id: str) -> str | None:
        return await self._render(owner_item_id, preview=False)

    async def _render(self, owner_item_id: str, *, preview: bool) -> str | None:
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg is not available.")
        conn, folder = self._store.conn(), self._store.folder()
        _, clips = resolve_timeline(conn, folder, owner_item_id)
        if not clips:
            return None
        settings = (mb.get_item(conn, owner_item_id).get("data") or {}).get("director") or _DEFAULT
        w, h, fps = settings["width"], settings["height"], settings["fps"]
        stamp = int(time.time() * 1000)
        if preview:
            rel = f"thumbs/director-{owner_item_id}-{stamp}.preview.mp4"
            cs = ComposeSettings(640, _even(640 * h / w), min(fps, 30), "ultrafast", 30,
                                 str(folder / rel))
        else:
            rel = f"thumbs/director-{owner_item_id}-export-{stamp}.mp4"
            cs = ComposeSettings(_even(w), _even(h), fps, "veryfast", 20, str(folder / rel))
        (folder / "thumbs").mkdir(parents=True, exist_ok=True)

        total = timeline_duration(clips)
        ok = await compose_render(
            build_compose_args(clips, cs), total,
            lambda f: self._events.broadcast(
                "events:timelineProgress", {"ownerItemId": owner_item_id, "fraction": f}
            ),
        )
        self._events.broadcast(
            "events:timelineProgress", {"ownerItemId": owner_item_id, "fraction": 1}
        )
        if not ok:
            if not preview:
                raise RuntimeError("Export render failed.")
            return None
        item = mb.get_item(conn, owner_item_id)
        field = "directorPreview" if preview else "directorExport"
        prev = item["data"].get(field)
        mb.update_item(conn, owner_item_id, {"data": {**item["data"], field: rel}})
        if prev and prev != rel:
            try:
                (folder / prev).unlink(missing_ok=True)
            except OSError:
                pass
        return rel


def export_frames(conn: Any, folder: Path) -> dict[str, Any]:
    """Copy each frame's hero-take Output, in order, into a numbered export dir (no browser
    folder picker, so it goes under the project's ``exports/<timestamp>/``)."""
    out_dir = folder / "exports" / str(int(time.time() * 1000))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        "SELECT s.name AS name, t.file_path AS file_path FROM frames s "
        "LEFT JOIN takes t ON s.hero_take_id = t.id ORDER BY s.position"
    ).fetchall()
    exported = 0
    skipped: list[str] = []
    for row in rows:
        if not row["file_path"]:
            skipped.append(row["name"])
            continue
        exported += 1
        ext = Path(row["file_path"]).suffix or ".png"
        safe = re.sub(r"[^\w.-]+", "_", row["name"])
        dest = out_dir / f"{str(exported).zfill(3)}_{safe}{ext}"
        shutil.copyfile(folder / row["file_path"], dest)
    return {"dir": str(out_dir), "exported": exported, "skipped": skipped}
