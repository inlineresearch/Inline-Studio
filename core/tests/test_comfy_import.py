from __future__ import annotations

from typing import Any

from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import parse_graph
from inline_core.graph.validate import validate
from inline_core.importer.comfy import import_comfy_prompt

# The canonical Comfy txt2img workflow (prompt/API format).
_PROMPT: dict[str, Any] = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
}


def _node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(n for n in graph["nodes"] if n["id"] == node_id)


def test_checkpoint_splits_into_three_loaders() -> None:
    graph = import_comfy_prompt(_PROMPT).graph
    types = {n["type"] for n in graph["nodes"]}
    assert {"load/diffusion-model", "load/text-encoder", "load/vae"} <= types
    assert _node(graph, "4:model")["params"]["file"] == "sdxl.safetensors"


def test_ksampler_wires_typed_edges_from_split_outputs() -> None:
    graph = import_comfy_prompt(_PROMPT).graph
    sample = _node(graph, "3")
    assert sample["type"] == "sample"
    assert sample["inputs"]["model"] == {"from": "4:model", "output": "model"}
    assert sample["inputs"]["positive"] == {"from": "6", "output": "conditioning"}
    assert sample["inputs"]["latent"] == {"from": "5", "output": "latent"}
    decode = _node(graph, "8")
    assert decode["inputs"]["vae"] == {"from": "4:vae", "output": "vae"}
    assert decode["inputs"]["latent"] == {"from": "3", "output": "latent"}


def test_prompt_text_becomes_an_input_text_node() -> None:
    graph = import_comfy_prompt(_PROMPT).graph
    text = _node(graph, "6:text")
    assert text["type"] == "input/text"
    assert text["params"]["text"] == "a fox"
    enc = _node(graph, "6")
    assert enc["inputs"]["prompt"] == {"from": "6:text", "output": "text"}
    assert enc["inputs"]["text_encoder"] == {"from": "4:clip", "output": "text_encoder"}


def test_sinks_ignored_unknown_reported() -> None:
    result = import_comfy_prompt(_PROMPT)
    assert result.skipped == []  # SaveImage is a known sink, not unmapped
    weird = import_comfy_prompt({"1": {"class_type": "SomeCustomNode", "inputs": {}}})
    assert weird.skipped == ["SomeCustomNode"]


def test_imported_graph_is_a_valid_core_graph() -> None:
    graph = import_comfy_prompt(_PROMPT).graph
    validate(parse_graph(graph), "8", build_default_registry())
