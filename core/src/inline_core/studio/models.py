"""Explicit, visible model downloads - the backend for the node's "missing models" popup.

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
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..models.requirements import RequirementsProvider, RequirementsRegistry


class ModelDownloads:
    """Answers "what's missing" and downloads components into the models dir on request.

    Model-agnostic: *what* a node needs comes from the ``RequirementsRegistry`` (built-ins register
    at boot, extensions when they load), and this class only knows how to report presence and move
    bytes. It contains no per-model knowledge.
    """

    def __init__(
        self,
        events: Any,
        on_change: Callable[[], None] | None = None,
        policy: Any = None,
        requirements: RequirementsRegistry | None = None,
    ) -> None:
        self._events = events
        self._on_change = on_change  # rescan the model catalog after a download lands
        self._policy = policy  # device policy, for the memory fit estimate (optional)
        self._requirements = requirements if requirements is not None else RequirementsRegistry()

    # --- requirements (the popup's data) --------------------------------------------------------

    def requirements(self, node_type: str) -> dict[str, Any]:
        """The node's model components with live presence + a memory fit estimate:
        ``{components: [...], allPresent, estimate}``.

        Returns an empty, all-present view for node types with no requirements (or when the model
        runtime isn't installed - the node then shows its own "unavailable" state instead)."""
        components = self._components(node_type)
        return {
            "components": [_component_json(c) for c in components],
            "allPresent": all(c.present for c in components),
            "estimate": self._estimate(node_type),
        }

    def _estimate(self, node_type: str) -> dict[str, Any] | None:
        """Whether the model will fit this machine - delegated to the node's provider, which owns
        the footprint knowledge. ``None`` whenever it can't be sized: a wrong estimate is worse
        than none, so the popup simply omits the warning."""
        provider = self._requirements.get(node_type)
        if provider is None or self._policy is None:
            return None
        try:
            return provider.estimate(self._policy)
        except Exception:  # noqa: BLE001 - a provider is extension code; never break the popup
            return None

    # --- download (explicit, user-triggered) ----------------------------------------------------

    def download(self, node_type: str, component_id: str) -> None:
        """Download one component (by id) or ``"all"`` missing ones, in a background thread."""
        loop = asyncio.get_running_loop()
        asyncio.create_task(asyncio.to_thread(self._run, node_type, component_id, loop))

    def _run(self, node_type: str, component_id: str, loop: asyncio.AbstractEventLoop) -> None:
        provider = self._requirements.get(node_type)
        if provider is None:
            return
        components = self._components(node_type)
        if component_id == "all":
            targets = [c for c in components if not c.present]
        else:
            targets = [c for c in components if c.id == component_id]
        for comp in targets:
            payload_id = comp.id
            try:
                self._download_component(
                    provider,
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
            except Exception as error:  # noqa: BLE001 - surface as a UI event, never crash the loop
                self._emit(
                    loop,
                    "events:modelDownloadError",
                    {"nodeType": node_type, "componentId": payload_id, "error": str(error)},
                )

    # --- internals ------------------------------------------------------------------------------

    def _components(self, node_type: str) -> list[Any]:
        """The node's components, or ``[]`` for a node type with no registered provider (which is
        also what a torch-less install sees - the node shows its own "unavailable" state)."""
        provider = self._requirements.get(node_type)
        if provider is None:
            return []
        try:
            return list(provider.components())
        except Exception:  # noqa: BLE001 - a provider is extension code; never break the popup
            return []

    def _download_component(
        self,
        provider: RequirementsProvider,
        comp: Any,
        on_progress: Callable[[float, str], None],
    ) -> None:
        """Fetch a component's single file from its repo into ``models/<category>/`` (flat, so the
        node's dropdown lists it). Downloads into a ``.part`` staging dir first, then moves the file
        into place under its basename, so a half-finished download never looks installed."""
        from huggingface_hub import HfApi, hf_hub_download

        category_dir: Path = provider.download_target(comp)
        staging = category_dir / (comp.filename + ".part")
        shutil.rmtree(staging, ignore_errors=True)

        files = _wanted_files(HfApi(), comp)
        total = sum(size for _, size in files)
        on_progress(0.0, f"Downloading {comp.label}…")

        # ``hf_hub_download`` blocks for the whole (multi-GB) file with no per-byte callback, so a
        # naive "emit after each file" jumps 0 → 99%. Instead, a watcher thread polls the staging
        # tree's byte size while the download runs and streams a real fraction. It only reports a
        # fraction when the repo metadata gave us a real total; otherwise size is unknown and we
        # leave the 0 → 1 endpoints (the status text still shows activity).
        stop = threading.Event()

        def _watch() -> None:
            while not stop.wait(0.5):
                if total <= 0:
                    continue
                got = _dir_size(staging)
                if got > 0:
                    on_progress(min(0.98, got / total), f"Downloading {comp.label}…")

        watcher = threading.Thread(target=_watch, name="model-download-progress", daemon=True)
        watcher.start()
        try:
            for rfilename, _size in files:
                hf_hub_download(comp.repo, rfilename, local_dir=str(staging))
        finally:
            stop.set()
            watcher.join(timeout=1.0)

        category_dir.mkdir(parents=True, exist_ok=True)
        for rfilename, _ in files:
            dest = category_dir / Path(rfilename).name
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
        "source": _source_label(component),
    }


def _source_label(component: Any) -> str:
    """Human-facing "which model" line: the repo + the exact file this component pulls (e.g.
    ``Comfy-Org/z_image/ae.safetensors``). Static - no network needed."""
    filename: str = getattr(component, "filename", "")
    if filename:
        return f"{component.repo}/{filename}"
    return component.repo


def _dir_size(path: Path) -> int:
    """Total bytes under ``path`` (recursively), for live download progress. Counts Hugging Face's
    ``.incomplete`` temp file as it grows, so the fraction advances during the blocking fetch.
    Best-effort: a file vanishing mid-walk (the final rename) is ignored."""
    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _wanted_files(api: Any, comp: Any) -> list[tuple[str, int]]:
    """(rfilename, size) for the single file this component pulls, sized from the repo metadata.

    Falls back to size 0 (progress by count) if the file isn't in the listing - the download still
    proceeds; ``hf_hub_download`` is the source of truth for whether the path exists."""
    info = api.model_info(comp.repo, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == comp.repo_file:
            return [(comp.repo_file, int(sibling.size or 0))]
    return [(comp.repo_file, 0)]
