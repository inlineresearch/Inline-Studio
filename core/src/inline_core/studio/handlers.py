"""Register the Studio InlineStudioApi channels natively on the RpcRouter, backed by ``StudioStore``
and the domain modules. This is the strangler-fig flip: once these are registered, the SPA's app
backend is Core (Python), not the legacy Node server.

Channel args arrive as a positional list (the ``{channel, args}`` wire shape). Each handler unpacks
them and returns the value to wrap in Ok. Not-yet-ported surfaces (generation, timeline,
export, embedded ComfyUI) register clear stubs so the UI degrades gracefully instead of a
"no handler".
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def register_studio_handlers(
    rpc: Any,
    store: StudioStore,
    *,
    core_models: Callable[[], dict[str, Any]],
    core_status: Callable[[], dict[str, Any]],
    generation: Any = None,
    fal_generation: Any = None,
    timeline: Any = None,
    app_version: str = "1.0.0",
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

    # --- project + app-global -------------------------------------------------------------------
    reg("project:create", lambda inp: store.create_project(inp["name"], inp.get("parentDir")))
    reg("project:open", lambda path: store.open_project(path))
    reg("project:openDialog", lambda: None)  # no native folder picker in a browser
    reg("project:openZip", lambda: None)
    reg("project:listRecent", store.list_recent)
    reg("project:current", store.current_project)
    reg("project:mediaDirs", store.media_dirs)
    reg("project:export", lambda _path: None)  # zip export: pending (see plan)
    reg("dialog:pickDirectory", lambda *_: str(cfg.workspace_dir()))
    reg("app:version", lambda: app_version)
    reg("settings:get", store.get_settings)
    reg("settings:setComfyUrl", store.set_comfy_url)
    reg("settings:setCoreUrl", store.set_core_url)
    reg("core:status", core_status)
    reg("core:models", core_models)

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
    reg("frames:unlink", lambda fid: fr.unlink_workflow(conn(), fid))
    reg("frames:setHero", lambda fid, take_id: fr.set_hero(conn(), fid, take_id))
    reg("frames:listTakes", lambda fid: fr.list_takes(conn(), fid))
    reg("frames:heroTakes", lambda: fr.hero_takes(conn()))
    reg("frames:listInputs", lambda: fr.list_inputs(conn()))
    reg("frames:addInput", lambda fid, aid: fr.add_input(conn(), fid, aid))
    reg("frames:addInputs", lambda fid, aids: fr.add_inputs(conn(), fid, aids))
    reg("frames:addSourceInput", lambda fid, src: fr.add_source_input(conn(), fid, src))
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
    reg("moodboard:list", lambda: mb.list_board(conn()))
    reg("moodboard:addAsset", lambda aid, x, y: mb.add_asset(conn(), aid, x, y))
    reg("moodboard:addText", lambda x, y: mb.add_text(conn(), x, y))
    reg("moodboard:addFrameFromAsset", lambda aid, x, y: mb.add_frame_from_asset(conn(), aid, x, y))
    reg("moodboard:addEmptyFrame", lambda x, y: mb.add_empty_frame(conn(), x, y))
    reg("moodboard:addFrameItem", lambda fid, x, y: mb.add_frame_item(conn(), fid, x, y))
    reg("moodboard:addPreview", lambda x, y: mb.add_preview(conn(), x, y))
    reg("moodboard:addLayer", lambda x, y: mb.add_layer(conn(), x, y))
    reg("moodboard:addDirector", lambda x, y: mb.add_director(conn(), x, y))
    reg("moodboard:addTrim", lambda x, y: mb.add_trim(conn(), x, y))
    reg("moodboard:addPrompt", lambda x, y: mb.add_prompt(conn(), x, y))
    reg("moodboard:addCoreNode", lambda t, x, y: mb.add_core_node(conn(), t, x, y))
    reg(
        "moodboard:addGenNode",
        lambda mid, x, y: mb.add_gen_node(conn(), mid, x, y, kind="image", params={}, title=mid),
    )
    reg("moodboard:updateItem", lambda iid, patch: mb.update_item(conn(), iid, patch))
    reg("moodboard:deleteItem", lambda iid: mb.delete_item(conn(), iid))
    reg("moodboard:importAndPlace", lambda _x, _y: [])  # browser uses /upload
    reg(
        "moodboard:createConnector",
        lambda f, t, sh=None, th=None: mb.create_connector(conn(), f, t, sh, th),
    )
    reg("moodboard:deleteConnector", lambda cid: mb.delete_connector(conn(), cid))
    reg("moodboard:setConnectorVolume", lambda cid, vol: mb.set_connector_volume(conn(), cid, vol))
    reg(
        "moodboard:replaceBoard",
        lambda items, connectors: mb.replace_board(conn(), items, connectors),
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
    reg("generation:resumePending", lambda: None)

    # --- fal settings (key stored server-side) --------------------------------------------------
    reg("falSettings:status", store.fal_status)
    reg("falSettings:setApiKey", store.set_fal_key)
    reg("falSettings:clearApiKey", store.clear_fal_key)
    reg("comfy:status", lambda: {"reachable": False, "url": ""})
    for ch in ("linkFrame", "uploadInputs", "pullWorkflow", "saveLiveWorkflow", "pushWorkflow",
               "pullLatest", "latestRun", "captureOutput"):
        reg(f"comfy:{ch}", not_wired("Embedded ComfyUI"))
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
