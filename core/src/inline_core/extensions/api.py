"""The extension author's public surface. Everything an extension imports lives here.

An extension ships one ``register(reg: ExtensionRegistrar)`` entry point that registers
``@inline_node``-decorated ``NodeRunner`` classes.

The decorator has no import-time side effect - it only attaches a descriptor. That is what lets the
same code load into a scratch registry during install validation and the live one on activation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..device.policy import DevicePolicy
from ..graph.descriptor import NodeDescriptor, ParamField, Port
from ..graph.registry import Registry
from ..graph.runners import NodeRunner
from ..graph.schema import PortKind
from ..media import MediaKind
from ..models.requirements import RequirementsProvider, RequirementsRegistry
from ..runtime.store import TakeStore

#: Attribute the decorator attaches to a runner class.
DESCRIPTOR_ATTR = "__inline_descriptor__"

#: ``inline_node`` takes a keyword argument named ``type`` to mirror ``NodeDescriptor.type``, which
#: shadows the builtin inside its body. Captured here so the runtime class check still works.
_type = type


class ExtensionError(RuntimeError):
    """An extension broke the contract. Fails that extension, never the server."""


def inline_node(
    *,
    type: str,  # noqa: A002 - mirrors NodeDescriptor.type
    title: str,
    category: str,
    inputs: Sequence[Port] = (),
    outputs: Sequence[Port] = (),
    params: Sequence[ParamField] = (),
    output_kind: MediaKind | None = None,
    icon: str = "",
    hidden: bool = False,
) -> Callable[[type[NodeRunner]], type[NodeRunner]]:
    """Attach a ``NodeDescriptor`` to a ``NodeRunner`` subclass. ``source`` is stamped by the
    registrar, not settable here, so provenance can't be spoofed."""

    def decorate(cls: type[NodeRunner]) -> type[NodeRunner]:
        # Checked at runtime because the annotation is a promise an author can break.
        if not (isinstance(cls, _type) and issubclass(cls, NodeRunner)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ExtensionError(f"@inline_node requires a NodeRunner subclass, got {cls!r}")
        setattr(
            cls,
            DESCRIPTOR_ATTR,
            NodeDescriptor(
                type=type,
                title=title,
                category=category,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                params=tuple(params),
                output_kind=output_kind,
                icon=icon,
                hidden=hidden,
            ),
        )
        return cls

    return decorate


def descriptor_of(cls: type[NodeRunner]) -> NodeDescriptor | None:
    descriptor = getattr(cls, DESCRIPTOR_ATTR, None)
    return descriptor if isinstance(descriptor, NodeDescriptor) else None


class ExtensionRegistrar:
    """Handed to an extension's ``register()``. The entire v1 capability surface.

    Namespacing is enforced here, not by convention: channels carry the ``ext:<id>:`` prefix, node
    types must be manifest-declared, and ``source`` is stamped."""

    def __init__(
        self,
        registry: Registry,
        extension_id: str,
        *,
        store: TakeStore,
        policy: DevicePolicy,
        requirements: RequirementsRegistry,
        data_root: Path,
        declared_nodes: Sequence[str] = (),
        enabled_nodes: Sequence[str] | None = None,
        rpc: Any = None,
        events: Any = None,
    ) -> None:
        self._registry = registry
        self._extension_id = extension_id
        self._store = store
        self._policy = policy
        self._requirements = requirements
        self._data_root = data_root
        self._declared = frozenset(declared_nodes)
        #: None means "register everything declared" (install-time validation).
        self._enabled = frozenset(enabled_nodes) if enabled_nodes is not None else None
        self._rpc = rpc
        self._events = events
        #: What actually landed in the registry, so disabling can be undone precisely.
        self.registered_nodes: list[str] = []
        self.registered_channels: list[str] = []
        #: Declared and offered by the author but switched off by the user.
        self.skipped_nodes: list[str] = []

    # --- nodes -----------------------------------------------------------------------------------

    def node(self, cls: type[NodeRunner], runner: NodeRunner | None = None) -> None:
        """Register one ``@inline_node``-decorated runner class.

        A node the user has switched off is validated and then skipped, so toggling it back on is
        just another ``register()`` call - never a restart.
        """
        descriptor = descriptor_of(cls)
        if descriptor is None:
            raise ExtensionError(f"{cls.__name__} is missing the @inline_node decorator")
        if self._declared and descriptor.type not in self._declared:
            raise ExtensionError(
                f"node type {descriptor.type!r} is not declared in the manifest's nodes[]"
            )
        _check_ports(descriptor)
        if self._enabled is not None and descriptor.type not in self._enabled:
            self.skipped_nodes.append(descriptor.type)
            return
        existing = self._registry.has(descriptor.type)
        if existing and not self._registry.get(descriptor.type).source.startswith("ext:"):
            raise ExtensionError(
                f"node type {descriptor.type!r} is already provided by Core and cannot be replaced"
            )
        stamped = replace(descriptor, source=f"ext:{self._extension_id}")
        self._registry.register(stamped, runner if runner is not None else cls())
        self.registered_nodes.append(descriptor.type)

    def nodes(self, *classes: type[NodeRunner]) -> None:
        for cls in classes:
            self.node(cls)

    # --- model requirements -----------------------------------------------------------------------

    def models(self, node_type: str, provider: RequirementsProvider) -> None:
        """Declare what a node needs on disk, so it flows through the existing download popup."""
        self._requirements.register(node_type, provider)

    # --- backend channels ------------------------------------------------------------------------

    def rpc_channel(self, method: str, fn: Callable[..., Any]) -> None:
        """Register ``ext:<id>:<method>``. The prefix is forced, so an author cannot shadow a
        core channel like ``project:open``."""
        if self._rpc is None:
            return
        import inspect

        channel = self.channel(method)

        async def handler(args: list[Any]) -> Any:
            result = fn(*args)
            if inspect.isawaitable(result):
                result = await result
            return result

        self._rpc.register(channel, handler)
        self.registered_channels.append(channel)

    def emit(self, event: str, payload: Any) -> None:
        """Broadcast ``ext:<extension>:<event>`` to connected clients."""
        if self._events is not None:
            self._events.broadcast(self.channel(event), payload)

    def channel(self, name: str) -> str:
        if not name or ":" in name:
            raise ExtensionError(f"channel name {name!r} must be a bare method name")
        return f"ext:{self._extension_id}:{name}"

    # --- engine handles --------------------------------------------------------------------------

    @property
    def takes(self) -> TakeStore:
        return self._store

    @property
    def device(self) -> DevicePolicy:
        """Placement policy. Never pick a device yourself - ask ``device.placement(role)``."""
        return self._policy

    @property
    def data_dir(self) -> Path:
        """This extension's private scratch dir. Writing elsewhere is a CRITICAL scanner finding."""
        path = self._data_root / self._extension_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _check_ports(descriptor: NodeDescriptor) -> None:
    """Reject custom port kinds: ``port_satisfies`` must stay total, or graph validation could no
    longer decide edge legality without running extension code."""
    for port in (*descriptor.inputs, *descriptor.outputs):
        if not isinstance(port.kind, PortKind):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ExtensionError(
                f"port {port.id!r} on {descriptor.type!r} uses an unsupported kind "
                f"{port.kind!r}; extensions must use the built-in PortKind values"
            )
