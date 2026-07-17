"""Best-effort ComfyUI workflow importer: maps Comfy's API/prompt format onto our primitive
vocabulary (not LiteGraph). Comfy's shape is `{nodeId: {class_type, inputs}}`.

Comfy inputs are either literals (widget values) or edges `[nodeId, outputIndex]`. Two structural
gaps we bridge: Comfy bundles model+clip+vae in one `CheckpointLoaderSimple` (we split it into three
`load/*` nodes), and puts prompt text in a widget (lifted into an `input/text` node). Only known
node types map; unknown ones are reported in `skipped`, never silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImportResult:
    graph: dict[str, Any]
    skipped: list[str] = field(default_factory=list)


@dataclass
class _Ctx:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    # (comfy_id, output_index) -> our edge {"from", "output"}
    outputs: dict[tuple[str, int], dict[str, str]] = field(default_factory=dict)
    # (our_node, our_input_port, comfy_edge) resolved after all nodes exist
    pending: list[tuple[dict[str, Any], str, tuple[str, int]]] = field(default_factory=list)


def _is_edge(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def _wire(node: dict[str, Any], port: str, value: Any, ctx: _Ctx) -> None:
    if _is_edge(value):
        ctx.pending.append((node, port, (value[0], int(value[1]))))


def _checkpoint(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    ckpt = inputs.get("ckpt_name", "")
    model_id, clip_id, vae_id = f"{comfy_id}:model", f"{comfy_id}:clip", f"{comfy_id}:vae"
    ctx.nodes.append({"id": model_id, "type": "load/diffusion-model", "params": {"file": ckpt}})
    ctx.nodes.append({"id": clip_id, "type": "load/text-encoder", "params": {"file": ckpt}})
    ctx.nodes.append({"id": vae_id, "type": "load/vae", "params": {"file": ckpt}})
    ctx.outputs[(comfy_id, 0)] = {"from": model_id, "output": "model"}
    ctx.outputs[(comfy_id, 1)] = {"from": clip_id, "output": "text_encoder"}
    ctx.outputs[(comfy_id, 2)] = {"from": vae_id, "output": "vae"}


def _clip_encode(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    text_id = f"{comfy_id}:text"
    ctx.nodes.append(
        {"id": text_id, "type": "input/text", "params": {"text": inputs.get("text", "")}}
    )
    node: dict[str, Any] = {
        "id": comfy_id,
        "type": "encode/text",
        "params": {},
        "inputs": {"prompt": {"from": text_id, "output": "text"}},
    }
    ctx.nodes.append(node)
    _wire(node, "text_encoder", inputs.get("clip"), ctx)
    ctx.outputs[(comfy_id, 0)] = {"from": comfy_id, "output": "conditioning"}


def _empty_latent(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    ctx.nodes.append(
        {
            "id": comfy_id,
            "type": "latent/empty",
            "params": {
                "width": inputs.get("width", 1024),
                "height": inputs.get("height", 1024),
                "batch": inputs.get("batch_size", 1),
            },
        }
    )
    ctx.outputs[(comfy_id, 0)] = {"from": comfy_id, "output": "latent"}


def _ksampler(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    node: dict[str, Any] = {
        "id": comfy_id,
        "type": "sample",
        "params": {
            "steps": inputs.get("steps", 20),
            "cfg": inputs.get("cfg", 5.0),
            "sampler": inputs.get("sampler_name", "euler"),
            "scheduler": inputs.get("scheduler", "simple"),
            "seed": inputs.get("seed", -1),
        },
        "inputs": {},
    }
    ctx.nodes.append(node)
    _wire(node, "model", inputs.get("model"), ctx)
    _wire(node, "positive", inputs.get("positive"), ctx)
    _wire(node, "negative", inputs.get("negative"), ctx)
    _wire(node, "latent", inputs.get("latent_image"), ctx)
    ctx.outputs[(comfy_id, 0)] = {"from": comfy_id, "output": "latent"}


def _vae_decode(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    node: dict[str, Any] = {"id": comfy_id, "type": "vae/decode", "params": {}, "inputs": {}}
    ctx.nodes.append(node)
    _wire(node, "latent", inputs.get("samples"), ctx)
    _wire(node, "vae", inputs.get("vae"), ctx)
    ctx.outputs[(comfy_id, 0)] = {"from": comfy_id, "output": "image"}


def _vae_encode(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    node: dict[str, Any] = {"id": comfy_id, "type": "vae/encode", "params": {}, "inputs": {}}
    ctx.nodes.append(node)
    _wire(node, "vae", inputs.get("vae"), ctx)
    _wire(node, "image", inputs.get("pixels"), ctx)
    ctx.outputs[(comfy_id, 0)] = {"from": comfy_id, "output": "latent"}


def _vae_loader(comfy_id: str, inputs: dict[str, Any], ctx: _Ctx) -> None:
    ctx.nodes.append(
        {"id": comfy_id, "type": "load/vae", "params": {"file": inputs.get("vae_name", "")}}
    )
    ctx.outputs[(comfy_id, 0)] = {"from": comfy_id, "output": "vae"}


_Handler = Callable[[str, dict[str, Any], _Ctx], None]

_HANDLERS: dict[str, _Handler] = {
    "CheckpointLoaderSimple": _checkpoint,
    "CLIPTextEncode": _clip_encode,
    "EmptyLatentImage": _empty_latent,
    "KSampler": _ksampler,
    "VAEDecode": _vae_decode,
    "VAEEncode": _vae_encode,
    "VAELoader": _vae_loader,
}

# Display sinks with no equivalent in our model (the decoded image is already the output/Frame).
_SINKS = {"SaveImage", "PreviewImage"}


def import_comfy_prompt(prompt: dict[str, Any]) -> ImportResult:
    """Convert a Comfy prompt-format workflow into our graph JSON (schemaVersion 1)."""
    ctx = _Ctx()
    skipped: list[str] = []
    for comfy_id, node in prompt.items():
        class_type = str(node.get("class_type", ""))
        if class_type in _SINKS:
            continue
        handler = _HANDLERS.get(class_type)
        if handler is None:
            skipped.append(class_type)
            continue
        handler(comfy_id, node.get("inputs", {}), ctx)

    for node, port, (src_id, src_index) in ctx.pending:
        edge = ctx.outputs.get((src_id, src_index))
        if edge is not None:
            node.setdefault("inputs", {})[port] = edge

    return ImportResult(graph={"schemaVersion": 1, "nodes": ctx.nodes}, skipped=skipped)
