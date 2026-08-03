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
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..models.requirements import RequirementsProvider, RequirementsRegistry

logger = logging.getLogger("inline_core.studio.models")


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
            # Suggested (optional) components never count as missing; control is opt-in.
            "allPresent": all(c.present for c in components if not getattr(c, "optional", False)),
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
            # "Download all" grabs required-missing only; a suggested component (e.g. a multi-GB
            # ControlNet) is fetched only when asked for by its own id.
            targets = [c for c in components if not c.present and not getattr(c, "optional", False)]
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
        into place under its basename, so a half-finished download never looks installed.

        Progress comes from huggingface_hub's own download counter via ``tqdm_class`` - real
        per-chunk motion, and it still resumes a partial file. (XET is disabled at process start so
        the plain HTTP path is used; XET reports nothing to tqdm.)"""
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import HfHubHTTPError

        category_dir: Path = provider.download_target(comp)
        staging = category_dir / (comp.filename + ".part")
        on_progress(0.0, f"Downloading {comp.label}…")
        tqdm_class = _progress_tqdm(on_progress, comp.label)
        folder = getattr(comp, "repo_folder", "")

        def _fetch(token: bool | None) -> str:
            if folder:
                # A component that is a directory - a sharded encoder, a tokenizer, a diffusers
                # checkpoint. Same staging and resume behaviour; only the fetch differs.
                root = snapshot_download(
                    comp.repo,
                    allow_patterns=f"{folder}/*",
                    local_dir=str(staging),
                    token=token,
                    tqdm_class=tqdm_class,
                )
                return str(Path(root) / folder)
            return hf_hub_download(
                comp.repo,
                comp.repo_file,
                local_dir=str(staging),
                token=token,
                tqdm_class=tqdm_class,
            )

        try:
            path = _fetch(None)  # ambient token, e.g. for a gated repo the user has access to
        except HfHubHTTPError as error:
            # A stale/invalid cached HF token 401s even on a public repo (HF masks it as "not
            # found"). Retry anonymously so a bad token never blocks a public model download.
            if getattr(getattr(error, "response", None), "status_code", None) not in (401, 403):
                raise
            shutil.rmtree(staging, ignore_errors=True)
            path = _fetch(False)

        category_dir.mkdir(parents=True, exist_ok=True)
        destination = category_dir / comp.filename
        # shutil.move drops a directory *inside* an existing destination rather than replacing it,
        # so a re-download would nest folders. Clear the target first, for files and folders alike.
        if destination.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
        elif destination.exists():
            destination.unlink()
        shutil.move(path, str(destination))
        shutil.rmtree(staging, ignore_errors=True)
        # Optional hook: a provider that needs to remember something about what just landed. H3
        # uses it to record which file is which partition, since the two are indistinguishable.
        after = getattr(provider, "after_download", None)
        if after is not None:
            try:
                after(comp, destination)
            except Exception:  # noqa: BLE001 - bookkeeping must never fail a completed download
                logger.warning("after_download hook failed for %s", comp.id)
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
        "optional": bool(getattr(component, "optional", False)),
    }


def _source_label(component: Any) -> str:
    """Human-facing "which model" line: the repo + the exact file this component pulls (e.g.
    ``Comfy-Org/z_image/ae.safetensors``). Static - no network needed."""
    filename: str = getattr(component, "filename", "")
    if filename:
        return f"{component.repo}/{filename}"
    return component.repo


def _progress_tqdm(on_progress: Callable[[float, str], None], label: str) -> Any:
    """A tqdm subclass handed to ``hf_hub_download`` as ``tqdm_class``; it forwards the download
    counter to ``on_progress``. It sums ``n`` itself because ``tqdm.update`` short-circuits (never
    touching ``self.n``) when its bar is disabled - which it is on a headless server. The 0.99 cap
    leaves the final 1.0 for the post-move ``ready``."""
    from tqdm.auto import tqdm

    class _Tqdm(tqdm):
        def update(self, n: float | None = 1) -> Any:
            done = super().update(n)
            self._forwarded = getattr(self, "_forwarded", 0.0) + (n or 0.0)
            if self.total:
                on_progress(min(0.99, self._forwarded / self.total), f"Downloading {label}…")
            return done

    return _Tqdm
