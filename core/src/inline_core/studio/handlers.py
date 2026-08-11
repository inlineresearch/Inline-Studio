"""Register the Studio InlineStudioApi channels natively on the RpcRouter, backed by ``StudioStore``
and the domain modules. This is the strangler-fig flip: once these are registered, the SPA's app
backend is Core (Python), not the legacy Node server.

Channel args arrive as a positional list (the ``{channel, args}`` wire shape). Each handler unpacks
them and returns the value to wrap in Ok. Not-yet-ported surfaces (generation, timeline,
export) register clear stubs so the UI degrades gracefully instead of a "no handler".
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import __version__
from . import assets as ax
from . import config as cfg
from . import fal as _fal
from . import frames as fr
from . import moodboard as mb
from .store import StudioStore


def _unlink(folder: Path, relatives: list[str]) -> None:
    for rel in relatives:
        try:
            (folder / rel).unlink(missing_ok=True)
        except OSError:
            pass



#: A node whose face is almost entirely an image preview. At the old 200x120 the render was a
#: thumbnail you had to open a lightbox to read, so generation nodes open large and portrait.
GENERATION_NODE_SIZE = (320, 480)
#: Loaders and other plumbing have no preview, so growing them would only cost canvas space.
COMPACT_NODE_SIZE = (200, 120)


def core_node_size(models: list[dict[str, Any]], core_type: str) -> tuple[int, int]:
    """How large a Core node opens. Decided here rather than in the renderer because only the
    registry knows whether a type produces a media output. An unknown type is assumed compact:
    guessing large would leave a big empty card on the canvas."""
    descriptor = next((m for m in models if m.get("type") == core_type), None)
    if descriptor and descriptor.get("outputKind"):
        return GENERATION_NODE_SIZE
    return COMPACT_NODE_SIZE


def register_studio_handlers(
    rpc: Any,
    store: StudioStore,
    *,
    core_models: Callable[[], dict[str, Any]],
    core_status: Callable[[], dict[str, Any]],
    generation: Any = None,
    fal_generation: Any = None,
    timeline: Any = None,
    training: Any = None,
    model_downloads: Any = None,
    activity: Any = None,
    model_tree: Callable[[], Any] | None = None,
    model_rescan: Callable[[], Any] | None = None,
    # Empty falls back to the installed package version; the launcher footer shows this.
    app_version: str = "",
) -> None:
    def reg(channel: str, fn: Callable[..., Any]) -> None:
        async def handler(args: list[Any]) -> Any:
            result = fn(*args)
            if inspect.isawaitable(result):
                result = await result  # async handlers (e.g. the ffmpeg timeline render)
            return result

        rpc.register(channel, handler)

    def conn() -> Any:
        return store.conn()

    def not_wired(feature: str) -> Callable[..., Any]:
        def fn(*_args: Any) -> Any:
            raise RuntimeError(f"{feature} isn't available on the single-process path yet.")

        return fn

    def _open_project(path: str) -> Any:
        opened = store.open_project(path)
        _settle_orphans()
        return opened

    def _current_project() -> Any:
        # Also the restore path: after a Core restart the client asks for the current project
        # rather than opening one, and its rows still need settling.
        current = store.current_project()
        _settle_orphans()
        return current

    def _settle_orphans() -> None:
        """A restart leaves rows stuck at running; settle them before anyone reads the history."""
        if activity is not None:
            activity.reconcile()

    # --- project + app-global -------------------------------------------------------------------
    reg("project:create", lambda inp: store.create_project(inp["name"], inp.get("parentDir")))
    reg("project:open", _open_project)
    reg("project:openDialog", lambda: None)  # no native folder picker in a browser
    reg("project:openZip", lambda: None)
    reg("project:listRecent", store.list_recent)
    reg("project:current", _current_project)
    reg("project:close", store.close_project)
    reg("project:mediaDirs", store.media_dirs)
    reg("project:export", lambda _path: None)  # zip export: pending (see plan)
    reg("dialog:pickDirectory", lambda *_: str(cfg.workspace_dir()))
    reg("app:version", lambda: app_version or __version__)
    reg("settings:get", store.get_settings)
    reg("settings:setCoreUrl", store.set_core_url)
    reg("core:status", core_status)
    reg("core:models", core_models)

    # --- model requirements + explicit downloads (the node's "missing models" popup) ------------
    if model_downloads is not None:
        reg("models:requirements", lambda node_type: model_downloads.requirements(node_type))
        reg("models:download", lambda node_type, cid: model_downloads.download(node_type, cid))
    else:
        reg("models:requirements", lambda _node_type: {"components": [], "allPresent": True})
        reg("models:download", not_wired("Model downloads"))

    # Read-only listing of every models root, for the Models side panel.
    reg("models:tree", model_tree if model_tree is not None else lambda: [])
    # Pick up weight files added or removed on disk since start.
    reg("models:rescan", model_rescan if model_rescan is not None else lambda: {})

    # --- folders --------------------------------------------------------------------------------
    reg("folders:list", lambda: ax.list_folders(conn()))
    reg("folders:create", lambda inp: ax.create_folder(conn(), inp["name"], inp.get("parentId")))
    reg("folders:rename", lambda fid, name: ax.rename_folder(conn(), fid, name))
    reg("folders:delete", lambda fid: ax.delete_folder(conn(), fid))

    # --- assets ---------------------------------------------------------------------------------
    reg("assets:list", lambda: ax.list_assets(conn()))
    reg("assets:importDialog", lambda _folder_id=None: [])  # browser uses /upload instead
    reg(
        "assets:importPaths",
        lambda paths, folder_id: [
            a
            for a in (ax.import_file(conn(), store.folder(), p, folder_id) for p in paths if p)
            if a
        ],
    )

    def delete_asset(asset_id: str) -> None:
        paths = ax.delete_asset(conn(), asset_id)
        _unlink(store.folder(), paths)

    reg("assets:delete", delete_asset)

    # --- frames + takes + inputs ----------------------------------------------------------------
    reg("frames:list", lambda: fr.list_frames(conn()))
    reg("frames:importAsFrames", lambda: [])  # browser has no import dialog
    reg("frames:addFromAsset", lambda asset_id: fr.add_from_asset(conn(), asset_id))
    reg("frames:rename", lambda fid, name: fr.rename_frame(conn(), fid, name))
    reg("frames:reorder", lambda ids: fr.reorder_frames(conn(), ids))
    reg("frames:clone", lambda fid: fr.clone_frame(conn(), fid))
    reg("frames:setHero", lambda fid, take_id: fr.set_hero(conn(), fid, take_id))
    reg("frames:listTakes", lambda fid: fr.list_takes(conn(), fid))
    reg("frames:heroTakes", lambda: fr.hero_takes(conn()))
    reg("frames:listInputs", lambda: fr.list_inputs(conn()))
    reg("frames:addInput", lambda fid, aid, handle=None: fr.add_input(conn(), fid, aid, handle))
    reg("frames:addInputs", lambda fid, aids, handle=None: fr.add_inputs(conn(), fid, aids, handle))
    reg(
        "frames:addSourceInput",
        lambda fid, src, handle=None: fr.add_source_input(conn(), fid, src, handle),
    )
    reg("frames:removeInput", lambda fid, aid: fr.remove_input(conn(), fid, aid))
    reg("frames:removeInputById", lambda fid, iid: fr.remove_input_by_id(conn(), fid, iid))
    reg("frames:reorderInputs", lambda fid, aids: fr.reorder_inputs(conn(), fid, aids))
    reg("frames:listAllTakes", lambda: fr.list_all_takes(conn()))
    reg("frames:setFalParams", lambda fid, params: fr.set_fal_params(conn(), fid, params))
    # Resolve a fal frame's inputs (media data URIs) + prompt so the browser can build the request.
    reg(
        "frames:resolveFalInputs",
        lambda fid: _fal.resolve_fal_inputs(conn(), store.folder(), fid),
    )
    # Fal model metadata (kind/params/title) lives studio-side; until the frontend sends it,
    # sensibly (kind image, empty params). The GenNode UI still renders params from its own def.
    reg("frames:setModel", lambda fid, mid: fr.set_model(conn(), fid, mid, "image", {}))
    reg(
        "frames:setProvider",
        lambda fid, provider, mid=None: fr.set_provider(
            conn(), fid, provider, mid, "image" if provider == "fal" else None, {}
        ),
    )

    def delete_frame(frame_id: str) -> None:
        fr.delete_frame(conn(), frame_id)

    reg("frames:delete", delete_frame)

    def delete_take(take_id: str) -> None:
        path = fr.delete_take(conn(), take_id)
        if path:
            _unlink(store.folder(), [path])

    reg("frames:deleteTake", delete_take)

    # --- moodboard ------------------------------------------------------------------------------
    # `surface` defaults to the Studio moodboard so existing callers are unchanged; the Trainer tab
    # passes "trainer" to get its own isolated canvas out of the same tables.
    reg("moodboard:list", lambda surface=mb.STUDIO_SURFACE: mb.list_board(conn(), surface))
    reg("moodboard:addAsset", lambda aid, x, y: mb.add_asset(conn(), aid, x, y))
    reg("moodboard:addText", lambda x, y: mb.add_text(conn(), x, y))
    reg("moodboard:addFrameFromAsset", lambda aid, x, y: mb.add_frame_from_asset(conn(), aid, x, y))
    reg("moodboard:addEmptyFrame", lambda x, y: mb.add_empty_frame(conn(), x, y))
    reg("moodboard:addFrameItem", lambda fid, x, y: mb.add_frame_item(conn(), fid, x, y))
    reg("moodboard:addPreview", lambda x, y: mb.add_preview(conn(), x, y))
    reg("moodboard:addLayer", lambda x, y: mb.add_layer(conn(), x, y))
    reg("moodboard:addDirector", lambda x, y: mb.add_director(conn(), x, y))
    reg("moodboard:addTrim", lambda x, y: mb.add_trim(conn(), x, y))
    reg("moodboard:addLoader", lambda x, y: mb.add_loader(conn(), x, y))
    reg("moodboard:addControlSpace", lambda x, y: mb.add_control_space(conn(), x, y))
    reg("moodboard:addPrompt", lambda x, y: mb.add_prompt(conn(), x, y))
    def _size(core_type: str) -> tuple[int, int]:
        try:
            return core_node_size(core_models().get("models", []), core_type)
        except Exception:  # noqa: BLE001 - a sizing hint must never block adding a node
            return COMPACT_NODE_SIZE

    reg("moodboard:addCoreNode", lambda t, x, y: mb.add_core_node(conn(), t, x, y, *_size(t)))
    reg(
        "moodboard:addGenNode",
        lambda mid, x, y: mb.add_gen_node(conn(), mid, x, y, kind="image", params={}, title=mid),
    )
    reg("moodboard:updateItem", lambda iid, patch: mb.update_item(conn(), iid, patch))
    reg("moodboard:deleteItem", lambda iid: mb.delete_item(conn(), iid))

    def remove_core_output(item_id: str, take_id: str) -> None:
        path = mb.remove_core_node_output(conn(), item_id, take_id)
        if path:
            _unlink(store.folder(), [path])

    reg("moodboard:removeCoreOutput", lambda iid, tid: remove_core_output(iid, tid))
    reg("moodboard:importAndPlace", lambda _x, _y: [])  # browser uses /upload
    reg(
        "moodboard:createConnector",
        lambda f, t, sh=None, th=None: mb.create_connector(conn(), f, t, sh, th),
    )
    reg("moodboard:deleteConnector", lambda cid: mb.delete_connector(conn(), cid))
    reg("moodboard:setConnectorVolume", lambda cid, vol: mb.set_connector_volume(conn(), cid, vol))
    reg(
        "moodboard:replaceBoard",
        lambda items, connectors, surface=mb.STUDIO_SURFACE: mb.replace_board(
            conn(), items, connectors, surface
        ),
    )
    # Trainer-canvas nodes (plus the shared read-only resource node, which either canvas can host).
    reg("moodboard:addTrainDataset", lambda x, y: mb.add_train_dataset(conn(), x, y))
    reg("moodboard:addCaption", lambda x, y: mb.add_caption(conn(), x, y))
    reg("moodboard:addTrainer", lambda x, y: mb.add_trainer(conn(), x, y))
    reg("moodboard:addLossGraph", lambda x, y: mb.add_loss_graph(conn(), x, y))
    reg(
        "moodboard:addResource",
        lambda x, y, surface=mb.STUDIO_SURFACE: mb.add_resource(conn(), x, y, surface),
    )

    # --- generation -----------------------------------------------------------------------------
    if generation is not None:
        reg("generation:runWorkflow", lambda item_id: generation.run_workflow(item_id))
    else:
        reg("generation:runWorkflow", not_wired("Core-node generation"))

    # Fal: the browser passes a prebuilt request {endpoint, body, outputKind}; Core runs it.
    if fal_generation is not None:
        reg("generation:run", lambda frame_id, request: fal_generation.run(frame_id, request))
    else:
        reg("generation:run", not_wired("Fal generation"))

    def cancel_generation(frame_id: str | None = None) -> None:
        if generation is not None:
            generation.cancel(frame_id)
        if fal_generation is not None:
            fal_generation.cancel(frame_id)

    reg("generation:cancel", cancel_generation)

    def active_generations() -> list[dict[str, Any]]:
        """What is still running, so a reloaded page rebuilds its queue instead of losing it."""
        out: list[dict[str, Any]] = []
        for source in (generation, fal_generation):
            if source is not None:
                out.extend(source.active())
        return out

    reg("generation:active", active_generations)
    reg("generation:resumePending", lambda: None)

    # --- activity -------------------------------------------------------------------------------
    if activity is not None:
        reg("activity:list", activity.snapshot)
        reg("activity:history", lambda limit=50: activity.history(limit))
        reg("activity:cancel", activity.cancel)
        reg("activity:clearHistory", activity.clear_history)
    else:
        reg("activity:list", lambda: [])
        reg("activity:history", lambda _limit=50: [])
        reg("activity:cancel", not_wired("Run activity"))
        reg("activity:clearHistory", not_wired("Run activity"))

    # --- LoRA training (dataset CRUD + the training run subprocess) ------------------------------
    if training is not None:
        reg("training:listDatasets", lambda: training.list_datasets())
        reg("training:createDataset", lambda inp: training.create_dataset(inp))
        reg("training:listItems", lambda did: training.list_items(did))
        reg("training:addItems", lambda did, aids: training.add_items(did, aids))
        reg("training:addFromPath", lambda did, path: training.add_from_path(did, path))
        reg("training:removeItem", lambda iid: training.remove_item(iid))
        reg("training:setCaption", lambda iid, cap: training.set_caption(iid, cap))
        reg("training:autoCaption",
            lambda did, overwrite, model=None: training.auto_caption(did, overwrite, model))
        reg("training:captioners", lambda: training.captioners())
        reg("training:listRuns", lambda: training.list_runs())
        reg("training:start", lambda did, hp: training.start(did, hp))
        reg("training:resume", lambda rid: training.resume(rid))
        reg("training:cancel", lambda rid: training.cancel(rid))
        reg("training:discard", lambda rid: training.discard(rid))
        reg("training:status", lambda rid: training.status(rid))
    else:
        for ch in ("listDatasets", "createDataset", "listItems", "addItems", "removeItem",
                   "setCaption", "autoCaption", "captioners", "listRuns", "start", "resume",
                   "cancel", "discard", "status"):
            reg(f"training:{ch}", not_wired("LoRA training"))

    # --- fal settings (key stored server-side) --------------------------------------------------
    reg("falSettings:status", store.fal_status)
    reg("falSettings:setApiKey", store.set_fal_key)
    reg("falSettings:clearApiKey", store.clear_fal_key)
    # --- timeline (director/trim/export via ffmpeg) + folder export -----------------------------
    if timeline is not None:
        from .timeline.render import export_frames

        reg("timeline:resolve", lambda oid: timeline.resolve(oid))
        reg("timeline:resolveTrim", lambda iid: timeline.resolve_trim(iid))
        reg("timeline:setVolumes", lambda oid, l1, l2: timeline.set_volumes(oid, l1, l2))
        reg("timeline:buildPreview", lambda oid: timeline.build_preview(oid))
        reg("timeline:export", lambda oid: timeline.export(oid))
        reg("export:exportFrames", lambda: export_frames(conn(), store.folder()))
    else:
        for ch in ("resolve", "resolveTrim", "setVolumes", "buildPreview", "export"):
            reg(f"timeline:{ch}", not_wired("The video timeline"))
        reg("export:exportFrames", lambda: None)

    reg("updates:check", lambda: None)
    reg("updates:quitAndInstall", lambda: None)
