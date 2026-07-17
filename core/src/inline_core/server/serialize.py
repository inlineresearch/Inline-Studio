"""Domain objects -> contract JSON (camelCase). The API boundary; keeps the domain types clean."""

from __future__ import annotations

from typing import Any

from ..graph.descriptor import NodeDescriptor, ParamField, Port
from ..models.catalog import ModelCatalog
from ..runtime.progress import (
    CancelledEvent,
    NodeDoneEvent,
    ProgressEvent,
    RunDoneEvent,
    RunEvent,
)
from ..runtime.run import NodeRuntimeState, RunState
from ..takes import Take


def port_json(port: Port) -> dict[str, Any]:
    return {"id": port.id, "label": port.label, "kind": port.kind.value, "required": port.required}


def param_json(field: ParamField, catalog: ModelCatalog | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": field.key,
        "label": field.label,
        "widget": field.widget.value,
        "default": field.default,
    }
    if field.min is not None:
        out["min"] = field.min
    if field.max is not None:
        out["max"] = field.max
    if field.step is not None:
        out["step"] = field.step
    options = [{"value": o.value, "label": o.label} for o in field.options]
    if field.options_from is not None:
        out["optionsFrom"] = field.options_from
        if catalog is not None:
            options += [{"value": f, "label": f} for f in catalog.list(field.options_from)]
    if options:
        out["options"] = options
    if field.advanced:
        out["advanced"] = True
    return out


def descriptor_json(
    descriptor: NodeDescriptor, catalog: ModelCatalog | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": descriptor.type,
        "title": descriptor.title,
        "category": descriptor.category,
        "icon": descriptor.icon,
        "source": descriptor.source,
        "outputKind": descriptor.output_kind.value if descriptor.output_kind else None,
        "inputs": [port_json(p) for p in descriptor.inputs],
        "outputs": [port_json(p) for p in descriptor.outputs],
        "params": [param_json(p, catalog) for p in descriptor.params],
    }
    if descriptor.hidden:
        out["hidden"] = True
    return out


def take_json(take: Take) -> dict[str, Any]:
    return {
        "id": take.id,
        "runId": take.run_id,
        "nodeId": take.node_id,
        "kind": take.kind.value,
        "uri": take.uri,
        "hash": take.hash,
        "params": take.params,
        "createdAt": take.created_at,
    }


def node_json(node: NodeRuntimeState) -> dict[str, Any]:
    out: dict[str, Any] = {"state": node.state.value, "fraction": node.fraction}
    if node.phase is not None:
        out["phase"] = node.phase
    if node.step is not None:
        out["step"] = node.step
    if node.step_count is not None:
        out["stepCount"] = node.step_count
    if node.status:
        out["status"] = node.status
    return out


def run_json(state: RunState) -> dict[str, Any]:
    return {
        "runId": state.run_id,
        "status": state.status.value,
        "target": state.target,
        "fraction": state.fraction,
        "nodes": {node_id: node_json(n) for node_id, n in state.nodes.items()},
        "takes": [take_json(t) for t in state.takes],
        "error": (
            {"nodeId": state.error.node_id, "message": state.error.message}
            if state.error is not None
            else None
        ),
    }


def event_json(event: RunEvent) -> dict[str, Any]:
    if isinstance(event, ProgressEvent):
        out: dict[str, Any] = {
            "type": "progress",
            "runId": event.run_id,
            "nodeId": event.node_id,
            "phase": event.phase.value,
            "fraction": event.fraction,
            "status": event.status,
        }
        if event.step is not None:
            out["step"] = event.step
        if event.step_count is not None:
            out["stepCount"] = event.step_count
        if event.eta_ms is not None:
            out["etaMs"] = event.eta_ms
        return out
    if isinstance(event, NodeDoneEvent):
        return {
            "type": "node_done",
            "runId": event.run_id,
            "nodeId": event.node_id,
            "cached": event.cached,
            "takes": [take_json(t) for t in event.takes],
        }
    if isinstance(event, RunDoneEvent):
        return {"type": "run_done", "runId": event.run_id}
    if isinstance(event, CancelledEvent):
        return {"type": "cancelled", "runId": event.run_id}
    return {
        "type": "error",
        "runId": event.run_id,
        "message": event.message,
        "nodeId": event.node_id,
    }
