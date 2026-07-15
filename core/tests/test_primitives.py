from __future__ import annotations

import pytest
from inline_core.errors import PortTypeError, UnknownNodeType
from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import parse_graph
from inline_core.graph.validate import validate


def _low_level_graph() -> object:
    # load/diffusion-model + load/vae + load/text-encoder -> encode -> sample -> vae/decode
    return parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "m", "type": "load/diffusion-model", "params": {"file": "z.safetensors"}},
                {"id": "v", "type": "load/vae", "params": {"file": "ae.safetensors"}},
                {"id": "te", "type": "load/text-encoder", "params": {"file": "qwen3"}},
                {"id": "p", "type": "input/text", "params": {"text": "a fox"}},
                {
                    "id": "enc",
                    "type": "encode/text",
                    "inputs": {
                        "text_encoder": {"from": "te", "output": "text_encoder"},
                        "prompt": {"from": "p", "output": "text"},
                    },
                },
                {"id": "lat", "type": "latent/empty", "params": {}},
                {
                    "id": "s",
                    "type": "sample",
                    "inputs": {
                        "model": {"from": "m", "output": "model"},
                        "positive": {"from": "enc", "output": "conditioning"},
                        "latent": {"from": "lat", "output": "latent"},
                    },
                },
                {
                    "id": "d",
                    "type": "vae/decode",
                    "inputs": {
                        "vae": {"from": "v", "output": "vae"},
                        "latent": {"from": "s", "output": "latent"},
                    },
                },
            ],
        }
    )


def test_primitives_are_served() -> None:
    types = {d.type for d in build_default_registry().descriptors()}
    assert {"load/diffusion-model", "encode/text", "sample", "vae/decode"} <= types


def test_low_level_graph_validates() -> None:
    validate(_low_level_graph(), "d", build_default_registry())


def test_wrong_engine_wire_is_rejected() -> None:
    # feed a VAE handle into sample's MODEL input
    graph = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "v", "type": "load/vae", "params": {}},
                {"id": "te", "type": "load/text-encoder", "params": {}},
                {"id": "p", "type": "input/text", "params": {"text": "x"}},
                {
                    "id": "enc",
                    "type": "encode/text",
                    "inputs": {
                        "text_encoder": {"from": "te", "output": "text_encoder"},
                        "prompt": {"from": "p", "output": "text"},
                    },
                },
                {"id": "lat", "type": "latent/empty", "params": {}},
                {
                    "id": "s",
                    "type": "sample",
                    "inputs": {
                        "model": {"from": "v", "output": "vae"},
                        "positive": {"from": "enc", "output": "conditioning"},
                        "latent": {"from": "lat", "output": "latent"},
                    },
                },
            ],
        }
    )
    with pytest.raises(PortTypeError):
        validate(graph, "s", build_default_registry())


def test_primitive_has_no_runner_until_c2() -> None:
    with pytest.raises(UnknownNodeType):
        build_default_registry().runner("sample")
