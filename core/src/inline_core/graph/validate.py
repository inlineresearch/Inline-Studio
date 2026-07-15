"""Up-front graph validation (contract section 3). Reject before running, never mid-graph."""

from __future__ import annotations

from ..errors import PortTypeError, UnknownNodeType
from .registry import Registry
from .schema import Graph, PortKind, port_satisfies
from .topo import topo_sort, upstream_closure


def validate(graph: Graph, target: str, registry: Registry) -> None:
    """Raise GraphValidationError (with node/port) on the first problem; return None if valid."""
    graph.node(target)

    for node in graph.nodes:
        if not registry.has(node.type):
            raise UnknownNodeType(f"Unknown node type {node.type!r}.", node_id=node.id)

    for node in graph.nodes:
        descriptor = registry.get(node.type)
        for port in descriptor.inputs:
            if port.required and port.id not in node.inputs:
                raise PortTypeError(
                    f"{descriptor.title} needs an input wired to {port.label!r}.",
                    node_id=node.id,
                    port=port.id,
                )
        for port_id, edges in node.inputs.items():
            in_port = descriptor.input(port_id)
            if in_port is None:
                raise PortTypeError(
                    f"{node.type!r} has no input port {port_id!r}.", node_id=node.id, port=port_id
                )
            if len(edges) > 1 and in_port.kind is not PortKind.IMAGE_LIST:
                raise PortTypeError(
                    f"Input {port_id!r} takes a single edge.", node_id=node.id, port=port_id
                )
            for edge in edges:
                source = graph.node(edge.from_node)
                out_port = registry.get(source.type).output(edge.output)
                if out_port is None:
                    raise PortTypeError(
                        f"{source.type!r} has no output port {edge.output!r}.",
                        node_id=node.id,
                        port=port_id,
                    )
                if not port_satisfies(out_port.kind, in_port.kind):
                    raise PortTypeError(
                        f"Cannot wire {out_port.kind.value} into {in_port.kind.value} "
                        f"at {node.id}.{port_id}.",
                        node_id=node.id,
                        port=port_id,
                    )

    closure = list(upstream_closure(target, graph.input_sources))
    topo_sort(closure, graph.input_sources)
