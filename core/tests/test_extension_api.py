"""The extension author's surface: the decorator and the registrar's enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inline_core.extensions.api import (
    ExtensionError,
    ExtensionRegistrar,
    descriptor_of,
    inline_node,
)
from inline_core.graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from inline_core.graph.registry import Registry
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind
from inline_core.models.requirements import RequirementsRegistry
from inline_core.server.rpc import EventBroadcaster, RpcRouter


@inline_node(
    type="acme/invert",
    title="Invert",
    category="Image",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("strength", "Strength", Widget.NUMBER, 1.0, min=0, max=1),),
)
class Invert(NodeRunner):
    produces_takes = False

    def run(self, node: Any, inputs: Any, ctx: Any) -> NodeResult:
        return NodeResult(outputs={})


def _registrar(
    registry: Registry,
    tmp_path: Path,
    *,
    declared: tuple[str, ...] = ("acme/invert",),
    enabled: tuple[str, ...] | None = None,
    rpc: RpcRouter | None = None,
    events: EventBroadcaster | None = None,
    requirements: RequirementsRegistry | None = None,
) -> ExtensionRegistrar:
    return ExtensionRegistrar(
        registry,
        "acme-tools",
        store=object(),  # pyright: ignore[reportArgumentType] - unused by these paths
        policy=object(),  # pyright: ignore[reportArgumentType]
        requirements=requirements or RequirementsRegistry(),
        data_root=tmp_path / "data",
        declared_nodes=declared,
        enabled_nodes=enabled,
        rpc=rpc,
        events=events,
    )


def test_decorator_attaches_a_descriptor_without_registering() -> None:
    """No import-time side effect: the same module must be loadable into a scratch registry during
    install validation and into the live one on activation."""
    descriptor = descriptor_of(Invert)
    assert descriptor is not None
    assert descriptor.type == "acme/invert"
    assert descriptor.output_kind is MediaKind.IMAGE
    assert descriptor.source == "builtin", "source is stamped by the registrar, not the author"


def test_decorator_rejects_a_non_runner_class() -> None:
    """Regression: `type` is a keyword argument of `inline_node`, so it shadows the builtin inside
    the decorator body. A naive `isinstance(cls, type)` compared against the node-type *string*
    and raised TypeError for every extension."""
    decorate = inline_node(type="acme/x", title="X", category="Test")

    class NotARunner:
        pass

    with pytest.raises(ExtensionError, match="requires a NodeRunner subclass"):
        decorate(NotARunner)  # pyright: ignore[reportArgumentType]


def test_registrar_stamps_provenance(tmp_path: Path) -> None:
    registry = Registry()
    _registrar(registry, tmp_path).nodes(Invert)
    assert registry.get("acme/invert").source == "ext:acme-tools"


def test_registrar_rejects_a_node_not_declared_in_the_manifest(tmp_path: Path) -> None:
    """Declared node types are what let preflight detect collisions before any extension code
    runs; a runner that registers something undeclared would defeat that."""
    registry = Registry()
    registrar = _registrar(registry, tmp_path, declared=("acme/something-else",))
    with pytest.raises(ExtensionError, match="not declared in the manifest"):
        registrar.node(Invert)
    assert not registry.has("acme/invert")


def test_registrar_refuses_to_replace_a_core_node(tmp_path: Path) -> None:
    registry = Registry()
    registry.register(
        NodeDescriptor(type="acme/invert", title="Core Thing", category="Generate")
    )
    with pytest.raises(ExtensionError, match="already provided by Core"):
        _registrar(registry, tmp_path).node(Invert)
    assert registry.get("acme/invert").title == "Core Thing"


def test_registrar_rejects_a_missing_decorator(tmp_path: Path) -> None:
    class Bare(NodeRunner):
        def run(self, node: Any, inputs: Any, ctx: Any) -> NodeResult:
            return NodeResult(outputs={})

    with pytest.raises(ExtensionError, match="missing the @inline_node decorator"):
        _registrar(registry := Registry(), tmp_path).node(Bare)
    assert registry.descriptors() == []


def test_registrar_rejects_a_custom_port_kind(tmp_path: Path) -> None:
    """port_satisfies has to stay total, so graph validation can decide edge legality without
    running extension code."""

    @inline_node(
        type="acme/custom",
        title="Custom",
        category="Test",
        outputs=(Port("out", "Out", "x/acme/pose"),),  # pyright: ignore[reportArgumentType]
    )
    class Custom(NodeRunner):
        def run(self, node: Any, inputs: Any, ctx: Any) -> NodeResult:
            return NodeResult(outputs={})

    registrar = _registrar(Registry(), tmp_path, declared=("acme/custom",))
    with pytest.raises(ExtensionError, match="unsupported kind"):
        registrar.node(Custom)


def test_rpc_channels_are_forced_into_the_pack_namespace(tmp_path: Path) -> None:
    """Without the forced prefix an extension could re-register `project:open` and silently
    intercept it."""
    rpc = RpcRouter()
    registrar = _registrar(Registry(), tmp_path, rpc=rpc)
    registrar.rpc_channel("listPresets", lambda: ["a"])
    assert rpc.has("ext:acme-tools:listPresets")
    assert registrar.registered_channels == ["ext:acme-tools:listPresets"]


def test_rpc_channel_rejects_a_qualified_name(tmp_path: Path) -> None:
    registrar = _registrar(Registry(), tmp_path, rpc=RpcRouter())
    with pytest.raises(ExtensionError, match="bare method name"):
        registrar.rpc_channel("project:open", lambda: None)


def test_emit_namespaces_the_event(tmp_path: Path) -> None:
    events = EventBroadcaster()
    queue = events.add()
    _registrar(Registry(), tmp_path, events=events).emit("done", {"ok": True})
    assert queue.get_nowait() == {"channel": "ext:acme-tools:done", "payload": {"ok": True}}


def test_data_dir_is_scoped_to_the_pack(tmp_path: Path) -> None:
    path = _registrar(Registry(), tmp_path).data_dir
    assert path.name == "acme-tools"
    assert path.is_dir()


def test_a_disabled_node_is_validated_then_skipped(tmp_path: Path) -> None:
    """The flattened model: everything imports, so toggling a node on later is just another
    register() call and never needs a restart."""
    registry = Registry()
    registrar = _registrar(registry, tmp_path, enabled=())

    registrar.node(Invert)

    assert not registry.has("acme/invert")
    assert registrar.skipped_nodes == ["acme/invert"]
    assert registrar.registered_nodes == []


def test_an_enabled_node_registers_normally(tmp_path: Path) -> None:
    registry = Registry()
    registrar = _registrar(registry, tmp_path, enabled=("acme/invert",))
    registrar.node(Invert)
    assert registry.has("acme/invert")
    assert registrar.skipped_nodes == []


def test_a_disabled_node_is_still_port_checked(tmp_path: Path) -> None:
    """Validation must not be skipped just because the node is off - otherwise a broken node only
    surfaces when a user switches it on."""

    @inline_node(
        type="acme/bad",
        title="Bad",
        category="Test",
        outputs=(Port("out", "Out", "x/custom"),),  # pyright: ignore[reportArgumentType]
    )
    class Bad(NodeRunner):
        def run(self, node: Any, inputs: Any, ctx: Any) -> NodeResult:
            return NodeResult(outputs={})

    registrar = _registrar(Registry(), tmp_path, declared=("acme/bad",), enabled=())
    with pytest.raises(ExtensionError, match="unsupported kind"):
        registrar.node(Bad)
