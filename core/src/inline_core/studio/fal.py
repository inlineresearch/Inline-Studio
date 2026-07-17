"""Fal generation on the single-process path — the server side of the fal relay.

The browser builds the fal request (endpoint + input body) from the studio-side node def, since fal
node definitions live there. Core owns the run: it submits to ``queue.fal.run`` with the key
(server-side only, never shipped to the page), polls to completion, parses the standard output
shapes, downloads the result into the project's ``takes/`` dir, and streams the generation events.

Ports the submit/poll/cancel logic of the Node ``electron/main/fal/client.ts``.
"""

from __future__ import annotations

import asyncio
import base64
import re
import uuid
from pathlib import Path
from typing import Any

from . import frames as fr
from . import moodboard as mb

_QUEUE_BASE = "https://queue.fal.run"

_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
}
_EXT_BY_KIND = {"image": ".png", "video": ".mp4", "audio": ".mp3"}


def _clamp01(n: float) -> float:
    return max(0.0, min(1.0, n))


# --- input resolution (frame inputs + prompt -> fal-usable data URIs) ----------------------------


def file_to_data_uri(abs_path: Path) -> str:
    mime = _MIME_BY_EXT.get(abs_path.suffix.lower(), "application/octet-stream")
    data = base64.b64encode(abs_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def resolve_fal_inputs(conn: Any, folder: Path, frame_id: str) -> dict[str, Any]:
    """A frame's inputs + prompt, resolved for the browser to build the fal request: media inputs as
    base64 data URIs grouped by kind, and the prompt text from a connected Prompt node."""
    images: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    for media in fr.frame_input_media(conn, frame_id):
        uri = file_to_data_uri(folder / media["filePath"])
        kind = media["kind"]
        (videos if kind == "video" else audios if kind == "audio" else images).append(uri)
    return {
        "images": images,
        "videos": videos,
        "audios": audios,
        "prompt": mb.prompt_text_for_frame(conn, frame_id),
    }


# --- output parsing (the standard fal response shapes) -------------------------------------------


def _ext_from(url: str, content_type: str | None, default: str) -> str:
    if content_type and "/" in content_type:
        sub = content_type.split("/")[-1].split(";")[0]
        mapped = {"jpeg": ".jpg", "quicktime": ".mov", "mpeg": ".mp3"}.get(sub)
        if mapped:
            return mapped
        if sub:
            return f".{sub}"
    match = re.search(r"\.[A-Za-z0-9]{1,5}(?:\?|$)", url)
    return match.group(0).split("?")[0] if match else default


def parse_outputs(response: Any, output_kind: str) -> list[dict[str, str]]:
    """Extract output refs from a fal response. Covers the shared shapes the node defs produce:
    ``{images:[{url,...}]}`` (image), ``{video:{url,...}}`` / ``{videos:[...]}`` (video/audio)."""
    if not isinstance(response, dict):
        return []
    refs: list[dict[str, str]] = []
    default_ext = _EXT_BY_KIND.get(output_kind, ".bin")
    if output_kind == "image":
        for img in response.get("images") or []:
            url = img.get("url") if isinstance(img, dict) else None
            if url:
                refs.append({
                    "url": url,
                    "ext": _ext_from(url, img.get("content_type"), default_ext),
                    "kind": "image",
                })
    else:
        single = response.get(output_kind)  # "video" | "audio"
        items = [single] if isinstance(single, dict) else (response.get(f"{output_kind}s") or [])
        for item in items:
            url = item.get("url") if isinstance(item, dict) else None
            if url:
                refs.append({
                    "url": url,
                    "ext": _ext_from(url, item.get("content_type"), default_ext),
                    "kind": output_kind,
                })
    return refs


# --- the fal HTTP client (submit / poll / cancel) ------------------------------------------------


def _resolve_queue_urls(endpoint: str, submitted: dict[str, Any]) -> dict[str, str]:
    request_id = submitted["request_id"]
    base = "/".join(endpoint.split("/")[:2])  # fal queue lives under the base app id (owner/app)
    status_url = submitted.get("status_url") or f"{_QUEUE_BASE}/{base}/requests/{request_id}/status"
    response_url = (
        submitted.get("response_url")
        or re.sub(r"/status(\?.*)?$", "", submitted.get("status_url") or "")
        or f"{_QUEUE_BASE}/{base}/requests/{request_id}"
    )
    return {"requestId": request_id, "statusUrl": status_url, "responseUrl": response_url}


def _progress_from_status(status: dict[str, Any]) -> tuple[float, str]:
    state = status.get("status")
    if state == "IN_QUEUE":
        pos = status.get("queue_position")
        return 0.05, f"Queued (#{pos})" if pos else "Queued"
    if state == "IN_PROGRESS":
        for log in reversed(status.get("logs") or []):
            msg = (log or {}).get("message") or ""
            step = re.search(r"(\d+)\s*(?:/|of)\s*(\d+)", msg, re.I)
            if step:
                cur, total = int(step.group(1)), int(step.group(2))
                if total > 0 and 0 <= cur <= total:
                    return _clamp01(0.1 + 0.85 * (cur / total)), f"Generating {cur}/{total}"
            pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", msg)
            if pct:
                p = min(100.0, float(pct.group(1)))
                return _clamp01(0.1 + 0.85 * (p / 100)), f"Generating {round(p)}%"
        return 0.5, "Generating"
    if state == "COMPLETED":
        return 1.0, "Done"
    return 0.1, str(state or "Working")


class FalGeneration:
    """Runs a browser-built fal request server-side and streams it as Studio generation events."""

    def __init__(self, store: Any, events: Any) -> None:
        self._store = store
        self._events = events
        self._active: dict[str, bool] = {}

    def run(self, frame_id: str, request: dict[str, Any]) -> None:
        key = self._store.fal_key()
        if not key:
            self._events.broadcast(
                "events:generationError",
                {"targetFrameId": frame_id, "error": "Add a fal API key in Settings to generate."},
            )
            return
        self._active[frame_id] = True
        asyncio.create_task(self._run(frame_id, request, key))

    def cancel(self, frame_id: str | None = None) -> None:
        for fid in [frame_id] if frame_id else list(self._active.keys()):
            self._active.pop(fid, None)

    async def _run(self, frame_id: str, request: dict[str, Any], key: str) -> None:
        import httpx

        endpoint = request["endpoint"]
        body = request.get("body") or request.get("input") or {}
        output_kind = request.get("outputKind") or "image"
        headers = {"Authorization": f"Key {key}"}
        try:
            self._events.broadcast(
                "events:generationProgress",
                {"frameId": frame_id, "fraction": 0.05, "status": "Queued"},
            )
            async with httpx.AsyncClient(timeout=600) as client:
                sub = await client.post(f"{_QUEUE_BASE}/{endpoint}", headers=headers, json=body)
                sub.raise_for_status()
                handle = _resolve_queue_urls(endpoint, sub.json())
                sep = "&" if "?" in handle["statusUrl"] else "?"
                status_url = handle["statusUrl"] + sep + "logs=1"
                while self._active.get(frame_id):
                    await asyncio.sleep(1.5)
                    res = await client.get(status_url, headers=headers)
                    if res.status_code >= 500:
                        continue
                    res.raise_for_status()
                    status = res.json()
                    fraction, label = _progress_from_status(status)
                    self._events.broadcast(
                        "events:generationProgress",
                        {"frameId": frame_id, "fraction": fraction, "status": label},
                    )
                    if status.get("status") == "COMPLETED":
                        break
                if not self._active.get(frame_id):
                    return  # cancelled
                result = await client.get(handle["responseUrl"], headers=headers)
                result.raise_for_status()
                refs = parse_outputs(result.json(), output_kind)
                if not refs:
                    raise RuntimeError("The model returned no output.")
                take_id = None
                for ref in refs:
                    take_id = await self._save(client, frame_id, ref, handle["requestId"], body)
                if take_id:
                    self._events.broadcast(
                        "events:generationNodeDone", {"frameId": frame_id, "takeId": take_id}
                    )
            self._events.broadcast("events:generationDone", {"targetFrameId": frame_id})
        except Exception as error:  # noqa: BLE001
            self._events.broadcast(
                "events:generationError", {"targetFrameId": frame_id, "error": str(error)}
            )
        finally:
            self._active.pop(frame_id, None)

    async def _save(
        self, client: Any, frame_id: str, ref: dict[str, str], request_id: str, params: dict
    ) -> str:
        folder: Path = self._store.folder()
        data = await client.get(ref["url"])
        data.raise_for_status()
        rel = f"takes/{uuid.uuid4()}{ref['ext']}"
        (folder / "takes").mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data.content)
        take = fr.add_take(
            self._store.conn(), frame_id, rel, ref["kind"], params, comfy_prompt_id=request_id
        )
        return take["id"]
