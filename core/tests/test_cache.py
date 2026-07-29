from __future__ import annotations

from typing import Any

from helpers import make_registry

from inline_core.graph.cache import (
    asset_content_hashes,
    is_cache_eligible,
    node_cache_key,
)
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


def test_control_wired_disables_cache() -> None:
    """A gen node with a control map wired re-runs every time (never cached), so iterating on a pose
    always applies the current control - even at a fixed seed."""
    registry = make_registry()
    with_control = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "c", "type": "input/image",
                 "params": {"asset": {"ref": "path", "path": "/c.png"}}},
                {"id": "m1", "type": "fake/model", "params": {"seed": 7},
                 "inputs": {"control": {"from": "c", "output": "image"}}},
            ],
        }
    )
    assert not is_cache_eligible(with_control.node("m1"), registry)
    # Same node, fixed seed, but no control wired -> eligible as before.
    assert is_cache_eligible(_graph({"seed": 7}).node("m1"), registry)


def _image_graph(path: str) -> Graph:
    """A gen node fed by an input/image source node pointing at `path` (mirrors graph_build)."""
    return parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "img", "type": "input/image",
                 "params": {"asset": {"ref": "path", "path": path}}},
                {"id": "m1", "type": "fake/model", "params": {"seed": 7},
                 "inputs": {"image": {"from": "img", "output": "image"}}},
            ],
        }
    )


def test_asset_content_hashes_hash_file_bytes(tmp_path) -> None:
    f = tmp_path / "map.png"
    f.write_bytes(b"first")
    g = _image_graph(str(f))
    h1 = asset_content_hashes(g)
    assert set(h1) == {"img"}
    f.write_bytes(b"second-different")
    assert asset_content_hashes(g)["img"] != h1["img"]  # same path, new bytes -> new hash
    # A missing file is skipped (its path still keys the node via params).
    assert asset_content_hashes(_image_graph(str(tmp_path / "gone.png"))) == {}


def test_cache_key_invalidates_when_a_control_map_is_rerendered(tmp_path) -> None:
    """A re-rendered control map at the SAME path must change the downstream gen node's cache key -
    otherwise a fixed-seed run serves a stale image (the reported bug)."""
    registry = make_registry()
    f = tmp_path / "pose.png"
    g = _image_graph(str(f))

    f.write_bytes(b"pose-A")
    key_a = node_cache_key(g, "m1", registry, asset_content_hashes(g))
    f.write_bytes(b"pose-B")
    key_b = node_cache_key(g, "m1", registry, asset_content_hashes(g))
    assert key_a != key_b

    # With the empty map (the old behaviour) the stale key would have collided.
    assert node_cache_key(g, "m1", registry, {}) == node_cache_key(g, "m1", registry, {})
