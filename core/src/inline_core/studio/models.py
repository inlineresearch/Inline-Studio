"""Explicit, visible model downloads — the backend for the node's "missing models" popup.

**This is the only place in the engine that fetches a model over the network.** The runner loads
everything ``local_files_only=True`` (never downloads); here the user explicitly asks for a
component and we pull it from the reference repo **straight into ``models/<category>/``** (never the
hidden HF cache), streaming progress over the ``/events`` socket. So a model arrives by exactly two
paths: the user drops files under ``models/``, or this downloader writes them there.

Fire-and-forget, mirroring ``CoreGeneration``: ``requirements`` answers synchronously; ``download``
schedules a background thread and returns immediately, emitting ``events:modelDownload*`` frames.
The blocking Hugging Face calls run off the event loop, so progress is marshalled back with
``loop.call_soon_threadsafe`` (``EventBroadcaster`` is not thread-safe).
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ZIMAGE_TYPE = "alibaba/z-image-turbo"
_SKIP_TOP_LEVEL = {".gitattributes", "readme.md", "license", "license.md", ".gitignore"}


class ModelDownloads:
    """Answers "what's missing" and downloads components into the models dir on request."""

    def __init__(self, events: Any, on_change: Callable[[], None] | None = None) -> None:
        self._events = events
        self._on_change = on_change  # rescan the model catalog after a download lands

    # --- requirements (the popup's data) --------------------------------------------------------

    def requirements(self, node_type: str) -> dict[str, Any]:
        """The node's model components with live presence. ``{components: [...], allPresent}``.

        Returns an empty, all-present view for node types with no requirements (or when the model
        runtime isn't installed — the node then shows its own "unavailable" state instead)."""
        components = self._components(node_type)
        return {
            "components": [_component_json(c) for c in components],
            "allPresent": all(c.present for c in components),
        }

    # --- download (explicit, user-triggered) ----------------------------------------------------

    def download(self, node_type: str, component_id: str) -> None:
        """Download one component (by id) or ``"all"`` missing ones, in a background thread."""
        loop = asyncio.get_running_loop()
        asyncio.create_task(asyncio.to_thread(self._run, node_type, component_id, loop))

    def _run(self, node_type: str, component_id: str, loop: asyncio.AbstractEventLoop) -> None:
        components = self._components(node_type)
        if component_id == "all":
            targets = [c for c in components if not c.present]
        else:
            targets = [c for c in components if c.id == component_id]
        for comp in targets:
            payload_id = comp.id
            try:
                self._download_component(
                    comp,
                    lambda frac, status, cid=payload_id: self._emit(
                        loop,
                        "events:modelDownloadProgress",
                        {
                            "nodeType": node_type,
                            "componentId": cid,
                            "fraction": frac,
                            "status": status,
                        },
                    ),
                )
                self._rescan()
                self._emit(
                    loop,
                    "events:modelDownloadDone",
                    {"nodeType": node_type, "componentId": payload_id},
                )
            except Exception as error:  # noqa: BLE001 — surface as a UI event, never crash the loop
                self._emit(
                    loop,
                    "events:modelDownloadError",
                    {"nodeType": node_type, "componentId": payload_id, "error": str(error)},
                )

    # --- internals ------------------------------------------------------------------------------

    def _components(self, node_type: str) -> list[Any]:
        if node_type != _ZIMAGE_TYPE:
            return []
        try:
            from ..models.zimage.requirements import zimage_requirements
        except ImportError:
            return []  # zimage runtime absent — the node shows "unavailable", no requirements
        return zimage_requirements()

    def _download_component(self, comp: Any, on_progress: Callable[[float, str], None]) -> None:
        """Fetch a component's files from its repo into ``models/<category>/…``, flattening any
        source subfolders. Downloads into a ``.part`` staging dir first, then moves into place, so a
        half-finished download never looks installed."""
        from huggingface_hub import HfApi, hf_hub_download

        from ..models.zimage.requirements import download_target

        target: Path = download_target(comp)
        staging = target.parent / (target.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)

        files = _wanted_files(HfApi(), comp)
        total = sum(size for _, size in files) or 1
        on_progress(0.0, f"Downloading {comp.label}…")

        downloaded = 0
        for rfilename, size in files:
            hf_hub_download(comp.repo, rfilename, local_dir=str(staging))
            downloaded += size
            on_progress(min(0.99, downloaded / total), f"Downloading {comp.label}…")

        target.mkdir(parents=True, exist_ok=True)
        for rfilename, _ in files:
            dest = target / _flatten_rel(rfilename, comp.subfolders)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / rfilename), str(dest))
        shutil.rmtree(staging, ignore_errors=True)
        on_progress(1.0, f"{comp.label} ready")

    def _rescan(self) -> None:
        """Refresh the catalog so /v1/models options + the registry version see the new files."""
        if self._on_change is not None:
            self._on_change()

    def _emit(self, loop: asyncio.AbstractEventLoop, channel: str, payload: Any) -> None:
        loop.call_soon_threadsafe(self._events.broadcast, channel, payload)


def _component_json(component: Any) -> dict[str, Any]:
    return {
        "id": component.id,
        "label": component.label,
        "category": component.category,
        "present": component.present,
        "localPath": component.local_path,
        "repo": component.repo,
    }


def _wanted_files(api: Any, comp: Any) -> list[tuple[str, int]]:
    """(rfilename, size) for the repo files this component needs — a subfolder subset, or the whole
    repo (minus boilerplate) when it has no subfolders."""
    info = api.model_info(comp.repo, files_metadata=True)
    out: list[tuple[str, int]] = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if comp.subfolders:
            if not any(name.startswith(sf + "/") for sf in comp.subfolders):
                continue
        elif "/" not in name and name.lower() in _SKIP_TOP_LEVEL:
            continue
        out.append((name, int(sibling.size or 0)))
    return out


def _flatten_rel(rfilename: str, subfolders: tuple[str, ...]) -> str:
    """Where a downloaded file lands under the target: subfolder components are flattened (strip the
    leading ``vae/`` etc.); a whole-repo (pipeline) download keeps its layout."""
    for sf in subfolders:
        prefix = sf + "/"
        if rfilename.startswith(prefix):
            return rfilename[len(prefix) :]
    return rfilename
