"""The node registry: descriptors served at /v1/models plus the runner behind each type."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from ..errors import UnknownNodeType
from .descriptor import NodeDescriptor
from .primitives import register_primitives
from .runners import IMAGE_INPUT, TEXT_INPUT, ImageInputRunner, NodeRunner, TextInputRunner


class Registry:
    def __init__(self) -> None:
        self._descriptors: dict[str, NodeDescriptor] = {}
        self._runners: dict[str, NodeRunner] = {}

    def register(self, descriptor: NodeDescriptor, runner: NodeRunner | None = None) -> None:
        """Register a node. A descriptor with no runner is served + validated but cannot run yet."""
        self._descriptors[descriptor.type] = descriptor
        if runner is not None:
            self._runners[descriptor.type] = runner

    def get(self, node_type: str) -> NodeDescriptor:
        descriptor = self._descriptors.get(node_type)
        if descriptor is None:
            raise UnknownNodeType(f"Unknown node type {node_type!r}.")
        return descriptor

    def has(self, node_type: str) -> bool:
        return node_type in self._descriptors

    def runner(self, node_type: str) -> NodeRunner:
        runner = self._runners.get(node_type)
        if runner is None:
            raise UnknownNodeType(f"No runner registered for {node_type!r}.")
        return runner

    def descriptors(self) -> list[NodeDescriptor]:
        return list(self._descriptors.values())

    def version(self) -> str:
        # TODO(phase1): fold resolved dynamic options into this so installing a model bumps it.
        payload = json.dumps(sorted(self._descriptors), separators=(",", ":"))
        return f"r_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def build_default_registry() -> Registry:
    """A registry with the built-in source nodes and the low-level primitive descriptors.

    Source nodes have runners; the primitives are descriptor-only until their runners land (C2).
    Both are marked hidden: the Studio drives text/image inputs through its own Prompt/library
    nodes, so these plumbing types stay runnable but never appear in the add-node menu.
    """
    registry = Registry()
    registry.register(replace(TEXT_INPUT, hidden=True), TextInputRunner())
    registry.register(replace(IMAGE_INPUT, hidden=True), ImageInputRunner())
    register_primitives(registry)
    return registry
