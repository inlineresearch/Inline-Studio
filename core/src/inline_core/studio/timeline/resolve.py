"""Resolve a director node's derived timeline from its wired connections — a port of the Studio
``electron/main/timeline/resolve.ts``. Video inputs use the ``vin-*`` handles, user audio (L2) the
``ain-*`` handles, in slot order. Each input resolves to a file (a frame's output or an asset), is
probed for duration, and laid out sequentially. Returns a display model + the ``ResolvedClip`` list
the compose engine renders. Filmstrip/waveform display extras are omitted (best-effort UI niceties).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import assets as ax
from .. import frames as fr
from .. import moodboard as mb
from .compose import ResolvedClip
from .ffmpeg import ffmpeg_available, probe_media

_STILL_SECONDS = 4.0
_VIDEO_PREFIX = "vin-"
_AUDIO_PREFIX = "ain-"


def _slot_index(handle: Any, prefix: str) -> int | None:
    if not isinstance(handle, str) or not handle.startswith(prefix):
        return None
    try:
        return int(handle[len(prefix):])
    except ValueError:
        return None


def _safe_frame_name(conn: Any, frame_id: str) -> str:
    try:
        return fr.get_frame(conn, frame_id)["name"]
    except ValueError:
        return "frame"


def _resolve_source_ref(
    conn: Any,
    from_item: dict[str, Any] | None,
    by_id: dict[str, dict[str, Any]],
    connectors: list[dict[str, Any]],
    depth: int = 0,
) -> dict[str, Any] | None:
    """A connector source item -> a media ref, walking frame/preview/trim/asset (trim attaches its
    in/out window). ``depth`` guards trim->trim cycles."""
    if from_item is None or depth > 8:
        return None
    kind = from_item["type"]
    if kind == "frame" and from_item.get("frameId"):
        out = fr.resolve_frame_file(conn, from_item["frameId"])
        if out is None:
            return None
        return {
            "sourceId": from_item["frameId"], "frameId": from_item["frameId"],
            "filePath": out["filePath"], "kind": out["kind"],
            "label": _safe_frame_name(conn, from_item["frameId"]), "trimIn": None, "trimOut": None,
        }
    if kind == "preview":
        feed = next((k for k in connectors if k["toItemId"] == from_item["id"]), None)
        feed_frame = by_id.get(feed["fromItemId"]) if feed else None
        if feed_frame and feed_frame["type"] == "frame" and feed_frame.get("frameId"):
            out = fr.resolve_frame_file(conn, feed_frame["frameId"])
            if out is None:
                return None
            return {
                "sourceId": feed_frame["frameId"], "frameId": feed_frame["frameId"],
                "filePath": out["filePath"], "kind": out["kind"],
                "label": _safe_frame_name(conn, feed_frame["frameId"]),
                "trimIn": None, "trimOut": None,
            }
        return None
    if kind == "trim":
        feed = next((k for k in connectors if k["toItemId"] == from_item["id"]), None)
        upstream = by_id.get(feed["fromItemId"]) if feed else None
        ref = _resolve_source_ref(conn, upstream, by_id, connectors, depth + 1)
        if ref is None:
            return None
        trim = (from_item.get("data") or {}).get("trim")
        return {**ref, "trimIn": trim["inPoint"] if trim else None,
                "trimOut": trim["outPoint"] if trim else None}
    if kind == "asset" and from_item.get("assetId"):
        asset = ax.asset_file(conn, from_item["assetId"])
        if asset is None:
            return None
        return {
            "sourceId": from_item["assetId"], "frameId": None, "filePath": asset["filePath"],
            "kind": asset["kind"], "label": asset["name"], "trimIn": None, "trimOut": None,
        }
    return None


def resolve_timeline(
    conn: Any, folder: Path, owner_item_id: str
) -> tuple[dict[str, Any], list[ResolvedClip]]:
    director = mb.get_item(conn, owner_item_id)
    data = director.get("data") or {}
    l1_volume = data.get("l1Volume", 1)
    l2_volume = data.get("l2Volume", 1)
    board = mb.list_board(conn)
    by_id = {i["id"]: i for i in board["items"]}
    incoming = [c for c in board["connectors"] if c["toItemId"] == owner_item_id]

    def pick(prefix: str) -> list[dict[str, Any]]:
        slotted = [
            (c, _slot_index((c.get("data") or {}).get("targetHandle"), prefix)) for c in incoming
        ]
        return [c for c, slot in sorted(
            (x for x in slotted if x[1] is not None), key=lambda x: x[1]
        )]

    display: dict[str, Any] = {"video": [], "l2": [], "l1Volume": l1_volume, "l2Volume": l2_volume}
    clips: list[ResolvedClip] = []

    def layout(conns: list[dict[str, Any]], track: int, volume: float) -> None:
        cursor = 0.0
        for conn_row in conns:
            ref = _resolve_source_ref(
                conn, by_id.get(conn_row["fromItemId"]), by_id, board["connectors"]
            )
            if ref is None:
                continue
            abs_path = folder / ref["filePath"]
            if not abs_path.is_file():
                continue
            if ref["kind"] == "image":
                probe = {"durationSec": _STILL_SECONDS, "hasAudio": False}
            elif ffmpeg_available():
                probe = probe_media(str(abs_path))
            else:
                probe = {"durationSec": _STILL_SECONDS, "hasAudio": ref["kind"] == "audio"}
            full = probe["durationSec"] if probe["durationSec"] > 0 else _STILL_SECONDS

            in_point, out_point = 0.0, full
            if ref["kind"] != "image" and (ref["trimIn"] is not None or ref["trimOut"] is not None):
                in_point = min(max(ref["trimIn"] or 0, 0), full)
                raw_out = ref["trimOut"] or 0
                out_point = min(raw_out, full) if raw_out > in_point else full
            duration = max(0.04, out_point - in_point)

            input_volume = (conn_row.get("data") or {}).get("volume", 1)
            if not isinstance(input_volume, (int, float)):
                input_volume = 1
            input_volume = min(1, max(0, input_volume))
            clip = {
                "key": ref["sourceId"], "connectorId": conn_row["id"], "volume": input_volume,
                "frameId": ref["frameId"], "label": ref["label"], "kind": ref["kind"],
                "startTime": cursor, "duration": duration, "audioPeaks": None,
                "peaksStart": 0, "peaksEnd": 1,
                "thumbnail": ref["filePath"] if ref["kind"] == "image" else None,
            }
            (display["video"] if track == 0 else display["l2"]).append(clip)
            clips.append(ResolvedClip(
                kind=ref["kind"], abs_path=str(abs_path), track=track, start_time=cursor,
                in_point=in_point, out_point=out_point, has_audio=bool(probe["hasAudio"]),
                volume=volume * input_volume,
            ))
            cursor += duration

    layout(pick(_VIDEO_PREFIX), 0, l1_volume)
    layout(pick(_AUDIO_PREFIX), 1, l2_volume)
    return display, clips


def resolve_trim(conn: Any, folder: Path, item_id: str) -> dict[str, Any] | None:
    board = mb.list_board(conn)
    by_id = {i["id"]: i for i in board["items"]}
    feed = next((k for k in board["connectors"] if k["toItemId"] == item_id), None)
    upstream = by_id.get(feed["fromItemId"]) if feed else None
    ref = _resolve_source_ref(conn, upstream, by_id, board["connectors"])
    if ref is None:
        return None
    abs_path = folder / ref["filePath"]
    if not abs_path.is_file():
        return None
    if ref["kind"] == "image":
        duration = _STILL_SECONDS
    elif ffmpeg_available():
        probe = probe_media(str(abs_path))
        duration = probe["durationSec"] if probe["durationSec"] > 0 else 0.0
    else:
        duration = 0.0
    return {
        "key": ref["sourceId"], "kind": ref["kind"], "label": ref["label"],
        "durationSec": duration, "mediaPath": ref["filePath"],
        "thumbnail": ref["filePath"] if ref["kind"] == "image" else None, "audioPeaks": None,
    }
