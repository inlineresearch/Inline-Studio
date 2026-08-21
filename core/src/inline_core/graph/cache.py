"""The content-hash node cache. Identity = (type, canonical params, upstream keys, asset content).

Determinism rules from docs/contract.md section 4, including seed-based cache eligibility.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from ..takes import Take
from .registry import Registry
from .schema import Graph, Node, PortKind


class NodeCache(ABC):
    @abstractmethod
    def get(self, key: str) -> list[Take] | None: ...

    @abstractmethod
    def put(self, key: str, takes: list[Take]) -> None: ...


class InMemoryCache(NodeCache):
    def __init__(self) -> None:
        self._store: dict[str, list[Take]] = {}

    def get(self, key: str) -> list[Take] | None:
        return self._store.get(key)

    def put(self, key: str, takes: list[Take]) -> None:
        self._store[key] = list(takes)


def _canonical_params(node: Node, registry: Registry) -> dict[str, Any]:
    merged = {**registry.get(node.type).defaults(), **node.params}
    return {key: merged[key] for key in sorted(merged)}


def is_cache_eligible(node: Node, registry: Registry) -> bool:
    """False when a control map is wired, or any seed param resolves to a negative (random) value.

    A node driven by a control map re-runs every time: the user iterates on the pose/depth and
    expects each run to apply the current control, so a cached take would read as "control not
    taking effect" (even a re-render at the same seed must re-apply it)."""
    descriptor = registry.get(node.type)
    for port in descriptor.inputs:
        if port.kind is PortKind.CONTROL and node.inputs.get(port.id):
            return False
    defaults = descriptor.defaults()
    for key in descriptor.seed_keys():
        value = node.params.get(key, defaults.get(key))
        try:
            if int(value) < 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def node_cache_key(
    graph: Graph,
    node_id: str,
    registry: Registry,
    asset_hashes: dict[str, str],
    _memo: dict[str, str] | None = None,
) -> str:
    """A stable content hash for a node's output. Asset refs contribute their byte hash."""
    memo = _memo if _memo is not None else {}
    if node_id in memo:
        return memo[node_id]
    node = graph.node(node_id)
    upstream = {
        port_id: [
            [edge.output, node_cache_key(graph, edge.from_node, registry, asset_hashes, memo)]
            for edge in edges
        ]
        for port_id, edges in sorted(node.inputs.items())
    }
    payload = {
        "type": node.type,
        "params": _canonical_params(node, registry),
        "inputs": upstream,
        "asset": asset_hashes.get(node_id),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    memo[node_id] = digest
    return digest


def asset_content_hashes(graph: Graph) -> dict[str, str]:
    """The byte hash of each file-backed source node's asset, keyed by node id. Feeds
    ``node_cache_key`` so the cache invalidates when a file's *content* changes even though its path
    did not (a re-rendered control map, an in-place-replaced input image). Only ``ref="path"`` refs
    are hashable; a missing file is skipped - its path still keys the node through its params."""
    import os

    hashes: dict[str, str] = {}
    for node in graph.nodes:
        character = _character_hash(node)
        if character is not None:
            hashes[node.id] = character
        asset = node.params.get("asset")
        if not isinstance(asset, dict) or asset.get("ref") != "path":
            continue
        path = asset.get("path")
        if isinstance(path, str) and os.path.isfile(path):
            hashes[node.id] = _file_hash(path)
    return hashes


def _character_hash(node: Node) -> str | None:
    """A picked character's byte hash. Editing a character in place leaves the filename it is
    picked by unchanged, so without this the cache serves a take of the previous face.

    Read from the node that names one - `character/load` - because a generation node takes its
    character by wire. A cache key folds in its upstream keys, so hashing it here invalidates
    everything downstream of it too.
    """
    chosen = node.params.get("file") if node.type == "character/load" else None
    if not isinstance(chosen, str) or not chosen.strip():
        return None
    try:
        from ..characters import library
    except ImportError:
        return None
    path = library.resolve(chosen)
    return _file_hash(str(path)) if path is not None else None


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
