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
from .schema import Graph, Node


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
    """False when any seed param resolves to a negative (random) value."""
    descriptor = registry.get(node.type)
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
