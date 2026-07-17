"""The typed graph: PortKind, Node, Edge, Graph, and the JSON parser. Graph schemaVersion 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..errors import GraphValidationError

SCHEMA_VERSION = 1


class PortKind(str, Enum):
    # media (cross the wire as takes / assets)
    IMAGE = "image"
    IMAGE_LIST = "image[]"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    MASK = "mask"
    # engine handles (opaque, passed between low-level nodes; never a take)
    MODEL = "model"
    VAE = "vae"
    TEXT_ENCODER = "text-encoder"
    CONDITIONING = "conditioning"
    LATENT = "latent"


def port_satisfies(source: PortKind, target: PortKind) -> bool:
    """Whether an output of `source` kind may feed an input of `target` kind."""
    if source == target:
        return True
    # a single image satisfies a list input (a one-element list)
    return source is PortKind.IMAGE and target is PortKind.IMAGE_LIST


@dataclass(frozen=True)
class Edge:
    from_node: str
    output: str


@dataclass
class Node:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, list[Edge]] = field(default_factory=dict)


@dataclass
class Graph:
    schema_version: int
    nodes: list[Node]
    _by_id: dict[str, Node] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {n.id: n for n in self.nodes}

    def node(self, node_id: str) -> Node:
        node = self._by_id.get(node_id)
        if node is None:
            raise GraphValidationError(f"No node with id {node_id!r}.", node_id=node_id)
        return node

    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def input_sources(self, node_id: str) -> list[str]:
        """Distinct upstream node ids feeding this node (the edges for topo sort)."""
        seen: list[str] = []
        for edges in self.node(node_id).inputs.values():
            for e in edges:
                if e.from_node not in seen:
                    seen.append(e.from_node)
        return seen


def _parse_edge(raw: Any) -> Edge:
    if not isinstance(raw, dict):
        raise GraphValidationError("An input edge must be an object with 'from' and 'output'.")
    frm = raw.get("from")
    out = raw.get("output")
    if not isinstance(frm, str) or not isinstance(out, str):
        raise GraphValidationError("An input edge needs string 'from' and 'output'.")
    return Edge(from_node=frm, output=out)


def _parse_inputs(raw: Any, node_id: str) -> dict[str, list[Edge]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise GraphValidationError("Node 'inputs' must be an object.", node_id=node_id)
    inputs: dict[str, list[Edge]] = {}
    for port, val in raw.items():
        inputs[str(port)] = [_parse_edge(e) for e in val] if isinstance(val, list) else [
            _parse_edge(val)
        ]
    return inputs


def _parse_node(raw: Any) -> Node:
    if not isinstance(raw, dict):
        raise GraphValidationError("Each node must be an object.")
    node_id = raw.get("id")
    node_type = raw.get("type")
    if not isinstance(node_id, str) or not isinstance(node_type, str):
        raise GraphValidationError("Each node needs a string 'id' and 'type'.")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise GraphValidationError("Node 'params' must be an object.", node_id=node_id)
    return Node(
        id=node_id,
        type=node_type,
        params={str(k): v for k, v in params.items()},
        inputs=_parse_inputs(raw.get("inputs"), node_id),
    )


def parse_graph(data: Any) -> Graph:
    """Parse the contract's graph JSON into a typed Graph. Raises GraphValidationError on shape."""
    if not isinstance(data, dict):
        raise GraphValidationError("Graph must be an object.")
    version = data.get("schemaVersion")
    if version != SCHEMA_VERSION:
        raise GraphValidationError(f"Unsupported graph schemaVersion: {version!r}.")
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list):
        raise GraphValidationError("Graph 'nodes' must be a list.")
    nodes = [_parse_node(n) for n in raw_nodes]
    ids = [n.id for n in nodes]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise GraphValidationError(f"Duplicate node ids: {dupes}.")
    return Graph(schema_version=version, nodes=nodes)
