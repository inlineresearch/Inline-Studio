"""The small Core changes the extension system rests on: content-addressed registry versions,
node unregistration, and exclusive RPC channel registration.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from inline_core.graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from inline_core.graph.registry import Registry
from inline_core.graph.schema import PortKind
from inline_core.server.rpc import RpcRouter

DESCRIPTOR = NodeDescriptor(
    type="acme/thing",
    title="Thing",
    category="Test",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("scale", "Scale", Widget.NUMBER, 2),),
)


def test_version_changes_when_a_param_default_changes() -> None:
    """The regression that motivated the fix: an extension upgrade can change params while keeping
    its node types identical. Hashing types alone left the /v1/models ETag stable and the client
    kept serving a stale catalog."""
    registry = Registry()
    registry.register(DESCRIPTOR)
    before = registry.version()

    registry.register(replace(DESCRIPTOR, params=(ParamField("scale", "Scale", Widget.NUMBER, 4),)))

    assert registry.version() != before


def test_version_changes_when_a_node_is_unregistered() -> None:
    registry = Registry()
    registry.register(DESCRIPTOR)
    with_node = registry.version()
    registry.unregister(DESCRIPTOR.type)
    assert registry.version() != with_node


def test_version_is_stable_across_insertion_order() -> None:
    other = replace(DESCRIPTOR, type="acme/other")
    first, second = Registry(), Registry()
    first.register(DESCRIPTOR)
    first.register(other)
    second.register(other)
    second.register(DESCRIPTOR)
    assert first.version() == second.version()


def test_version_tracks_source_so_provenance_changes_bust_the_cache() -> None:
    registry = Registry()
    registry.register(DESCRIPTOR)
    builtin = registry.version()
    registry.register(replace(DESCRIPTOR, source="ext:acme:basic"))
    assert registry.version() != builtin


def test_unregister_removes_the_descriptor_and_its_runner() -> None:
    registry = Registry()
    registry.register(DESCRIPTOR)
    assert registry.has(DESCRIPTOR.type)
    registry.unregister(DESCRIPTOR.type)
    assert not registry.has(DESCRIPTOR.type)
    assert DESCRIPTOR.type not in [d.type for d in registry.descriptors()]


def test_unregister_is_a_no_op_for_unknown_types() -> None:
    Registry().unregister("nobody/home")


async def _handler(_args: list[Any]) -> str:
    return "ok"


def test_registering_an_existing_channel_raises() -> None:
    """Without this guard an extension could re-register `project:open` or `settings:get` and
    silently intercept every call to it - RpcRouter was last-write-wins."""
    router = RpcRouter()
    router.register("project:open", _handler)
    with pytest.raises(ValueError, match="already registered"):
        router.register("project:open", _handler)


def test_unregister_frees_the_channel_for_reuse() -> None:
    router = RpcRouter()
    router.register("ext:acme:go", _handler)
    router.unregister("ext:acme:go")
    assert not router.has("ext:acme:go")
    router.register("ext:acme:go", _handler)  # re-registration after disable must work
    assert router.has("ext:acme:go")
