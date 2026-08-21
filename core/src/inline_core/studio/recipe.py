"""Author a reproducible recipe for a generated take: the params it used, its prompt, and the raw
canvas subgraph (items + connectors of the target's upstream closure) so the image can rebuild the
graph that made it when dropped back in - even in someone else's project.

Reuses graph_build's closure walk. Unlike ``build_workflow_graph`` this keeps the *raw moodboard*
items (type / minimal data / position), not the flattened engine graph, because the client rebuilds
canvas nodes from it. Per-item data is trimmed to the rebuild-relevant fields so the recipe stays
small and never nests another node's take history (which itself carries recipes).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from . import frames as fr
from . import moodboard as mb
from .graph_build import _upstream_closure

logger = logging.getLogger("inline_core.recipe")

#: 2 typed every param and moved each node's models beside it; 1 was a flat param map.
RECIPE_VERSION = 2

#: node type -> the values a run would use where the item stores none. Set once by the server,
#: which is where the registry and the requirements providers are both in scope.
_param_fallbacks: Any = None


def set_param_resolver(resolve: Any) -> None:
    """Teach recipes what a node resolves to. Without it they record only what the user typed."""
    global _param_fallbacks
    _param_fallbacks = resolve


#: node type -> the models it needs but never names in a param. Set alongside the resolver above.
_node_models: Any = None


def set_model_resolver(resolve: Any) -> None:
    """Teach recipes what a node needs on disk beyond its own params."""
    global _node_models
    _node_models = resolve


#: node type -> {param key: kind}, so an exported param says what it is. Set by the server.
_param_kinds: Any = None


def set_kind_resolver(resolve: Any) -> None:
    """Teach recipes what each of a node's params is."""
    global _param_kinds
    _param_kinds = resolve


def _typed_params(
    node_type: str, params: dict[str, Any], wired: frozenset[str] = frozenset()
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Each param as {type, value}, plus the download coordinates for the ones naming a model.

    Two things a bare value cannot say: what it is, and where it comes from. Without the first a
    reader guesses from the param's name; without the second an export is only fetchable on a
    machine whose registry still carries the same entry.
    """
    kinds: dict[str, str] = {}
    if _param_kinds is not None and node_type:
        try:
            kinds = _param_kinds(node_type)
        except Exception:  # noqa: BLE001 - a recipe is metadata; never fail a render over it
            logger.warning("Could not resolve param kinds for %s", node_type)

    typed = {k: {"type": kinds.get(k, "string"), "value": v} for k, v in params.items()}

    # What the params name, plus what the node needs without naming it: LTX-2.5's duration head is
    # required and appears in no param, so only the node's own declaration can carry it.
    wanted: dict[str, str] = {}
    for key, value in params.items():
        # A wired `model`/`vae`/`text_encoder` is ignored at run time and the loader driving it
        # names its own file, so counting this one too listed the wrong checkpoint beside the right.
        if kinds.get(key) in ("model", "character") and str(value) and key not in wired:
            wanted.setdefault(str(value).replace("\\", "/").rsplit("/", 1)[-1], "")
    # A node's declared requirements name its *default* build, so a node set to klein-9b would
    # otherwise export klein-4b beside it. Whatever a param already named speaks for its folder.
    covered = _folders_of(list(wanted))
    for filename, category in _declared(node_type, params):
        if category in covered:
            continue
        wanted[filename] = category
    return typed, _coordinates(wanted)


def _folders_of(names: list[str]) -> set[str]:
    if not names:
        return set()
    from ..models import registry_index as ri

    try:
        return {v.get("directory", "") for v in ri.coordinates(names).values()}
    except Exception:  # noqa: BLE001 - an unreachable registry must not fail a render
        return set()


def _declared(node_type: str, params: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    if _node_models is None or not node_type:
        return []
    try:
        return list(_node_models(node_type, params or {}))
    except Exception:  # noqa: BLE001 - a recipe is metadata; never fail a render over it
        logger.warning("Could not resolve required models for %s", node_type)
        return []


def _coordinates(wanted: dict[str, str]) -> list[dict[str, str]]:
    """Each model as {directory, name, url}. The node's own category wins: the registry may not
    carry the file at all, and the folder it belongs in is still known."""
    if not wanted:
        return []
    from ..models import registry_index as ri

    try:
        found = ri.coordinates(list(wanted))
    except Exception:  # noqa: BLE001 - an unreachable registry must not fail a render
        found = {}
    return [
        {
            "directory": category or found.get(name, {}).get("directory", ""),
            "name": name,
            "url": found.get(name, {}).get("url", ""),
        }
        for name, category in wanted.items()
    ]


def _effective_params(core: dict[str, Any]) -> dict[str, Any]:
    """The params a run would actually use.

    A picker left alone stores nothing: the node shows the file the engine resolved and the run
    loads it, but the item's params stay empty. Recorded as-is, a recipe naming no weight cannot
    rebuild the same image on another machine, and reports needing no models at all.
    """
    stored = core.get("params") or {}
    node_type = str(core.get("type") or "")
    out: dict[str, Any] = {}
    if _param_fallbacks is not None and node_type:
        try:
            out.update(_param_fallbacks(node_type))
        except Exception:  # noqa: BLE001 - a recipe is metadata; never fail a render over it
            logger.warning("Could not resolve params for %s", node_type)
    out.update({k: v for k, v in stored.items() if v not in (None, "")})
    return out


def _item_data(
    conn: sqlite3.Connection, item: dict[str, Any], wired: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """The rebuild-relevant data for a recipe item. A fal gen node is a `frame` whose model + params
    live in the frames table, so it's looked up; everything else trims its own `data`."""
    if item["type"] == "frame" and item.get("frameId"):
        frame = fr.get_frame(conn, item["frameId"])
        if frame.get("provider") == "fal" and frame.get("modelId"):
            return {"fal": {"modelId": frame["modelId"], "params": frame.get("params") or {}}}
        return {}  # a rendered-frame image source - rebuilt structurally (media doesn't transfer)
    return _clean_data(item, wired)


def _wired_handles(item_id: str, connectors: list[dict[str, Any]]) -> frozenset[str]:
    """Input ports of ``item_id`` that a wire drives, so a param sharing the name is not exported
    as a model the graph needs: the loader on the other end already names the file."""
    return frozenset(
        handle
        for c in connectors
        if c["toItemId"] == item_id and (handle := (c.get("data") or {}).get("targetHandle"))
    )


def _clean_data(item: dict[str, Any], wired: frozenset[str] = frozenset()) -> dict[str, Any]:
    """The rebuild-relevant subset of an item's data - never the take history (`core.outputs`)."""
    kind = item["type"]
    data = item.get("data") or {}
    if kind == "core":
        core = data.get("core") or {}
        node_type = str(core.get("type") or "")
        params = _effective_params(core)
        typed, models = _typed_params(node_type, params, wired)
        entry: dict[str, Any] = {"type": node_type, "params": typed}
        if models:
            entry["models"] = models
        return {"core": entry}
    if kind == "prompt":
        return {"promptText": data.get("promptText") or ""}
    # Training nodes keep their settings, never their bindings: a dataset and a run are rows in
    # this project's database, so they name nothing in the project the recipe lands in.
    if kind == "train/lora":
        hyperparams = data.get("hyperparams") or {}
        # The base checkpoint follows the architecture, and no param on the node names the file, so
        # without this a published training graph asked the reader to work out a 26 GB download.
        models = _coordinates(dict(_declared(kind, {"hyperparams": hyperparams})))
        out: dict[str, Any] = {"hyperparams": hyperparams}
        if models:
            out["models"] = models
        return out
    if kind == "train/caption":
        return {
            "overwrite": bool(data.get("overwrite") or False),
            "captioner": data.get("captioner") or "",
        }
    if kind in ("train/dataset", "train/loss", "resource"):
        return {}
    if kind == "controlSpace":
        out: dict[str, Any] = {}
        if data.get("controlAssetId"):
            out["controlAssetId"] = data["controlAssetId"]
        if data.get("controlScene"):
            out["controlScene"] = data["controlScene"]
        return out
    if kind == "loader":
        return {"assetIds": data.get("assetIds") or []}
    if kind in ("text", "layer", "director", "trim"):
        return data  # small styling/label data, kept verbatim
    return {}


def _prompt_for(target_id: str, connectors: list[dict[str, Any]], items: dict[str, Any]) -> str:
    """The text of the prompt node wired into the target's ``prompt`` input (for quick display)."""
    for c in connectors:
        if c["toItemId"] == target_id and (c.get("data") or {}).get("targetHandle") == "prompt":
            src = items.get(c["fromItemId"])
            if src and src["type"] == "prompt":
                return str((src.get("data") or {}).get("promptText") or "")
    return ""


def build_recipe(conn: sqlite3.Connection, target_item_id: str) -> dict[str, Any]:
    """The recipe for the take produced by ``target_item_id`` (a canvas item id == node id)."""
    board = mb.list_board(conn)
    items = {i["id"]: i for i in board["items"]}
    connectors = board["connectors"]
    closure = _upstream_closure(target_item_id, connectors)

    recipe_items = [
        {
            "id": item["id"],
            "type": item["type"],
            "data": _item_data(conn, item, _wired_handles(node_id, connectors)),
            "x": item["x"],
            "y": item["y"],
            "width": item["width"],
            "height": item["height"],
            "assetId": item.get("assetId"),
            "frameId": item.get("frameId"),
        }
        for node_id in closure
        if (item := items.get(node_id)) is not None
    ]
    recipe_connectors = [
        {
            "fromItemId": c["fromItemId"],
            "toItemId": c["toItemId"],
            "data": c.get("data") or {},
        }
        for c in connectors
        if c["fromItemId"] in closure and c["toItemId"] in closure
    ]

    target = items.get(target_item_id) or {}
    core = (target.get("data") or {}).get("core") or {}
    return {
        "version": RECIPE_VERSION,
        "app": "inline-studio",
        "target": target_item_id,
        "coreType": core.get("type"),
        "params": core.get("params") or {},
        "prompt": _prompt_for(target_item_id, connectors, items),
        "graph": {"items": recipe_items, "connectors": recipe_connectors},
    }
