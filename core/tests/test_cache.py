from __future__ import annotations

from typing import Any

from helpers import make_registry
from inline_core.graph.cache import is_cache_eligible, node_cache_key
from inline_core.graph.schema import Graph, parse_graph


def _graph(model_params: dict[str, Any], prompt: str = "a fox") -> Graph:
    return parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "p1", "type": "input/text", "params": {"text": prompt}},
                {
                    "id": "m1",
                    "type": "fake/model",
                    "params": model_params,
                    "inputs": {"prompt": {"from": "p1", "output": "text"}},
                },
            ],
        }
    )


def test_cache_key_is_stable_and_param_sensitive() -> None:
    registry = make_registry()
    k1 = node_cache_key(_graph({"seed": 7, "steps": 8}), "m1", registry, {})
    k2 = node_cache_key(_graph({"steps": 8, "seed": 7}), "m1", registry, {})
    k3 = node_cache_key(_graph({"seed": 8, "steps": 8}), "m1", registry, {})
    assert k1 == k2
    assert k1 != k3


def test_cache_key_tracks_upstream_prompt() -> None:
    registry = make_registry()
    a = node_cache_key(_graph({"seed": 7}, prompt="a fox"), "m1", registry, {})
    b = node_cache_key(_graph({"seed": 7}, prompt="a cat"), "m1", registry, {})
    assert a != b


def test_seed_eligibility() -> None:
    registry = make_registry()
    assert is_cache_eligible(_graph({"seed": 7}).node("m1"), registry)
    assert not is_cache_eligible(_graph({"seed": -1}).node("m1"), registry)
    assert not is_cache_eligible(_graph({}).node("m1"), registry)
