"""``output_kind`` is the contract ``studio.generation._save_take`` routes a take on.

Before it did that, a descriptor whose ``output_kind`` disagreed with the take its runner returns
was harmless. Now it decides which take claims the node's canvas output, so a disagreement means a
node silently stops showing its result. These are the static halves of that invariant, checked
across the whole registry rather than per model, so a new node cannot land holding it wrong.
"""

from __future__ import annotations

from pathlib import Path

from inline_core.device.memory import MemoryPolicy
from inline_core.graph.registry import Registry, build_default_registry
from inline_core.graph.schema import PortKind, port_satisfies
from inline_core.media import MediaKind
from inline_core.runtime.file_store import FileTakeStore
from inline_core.server.bootstrap import register_models

#: The port kind a media output is natively carried on. A node may declare a narrower port that this
#: one satisfies - `control/apply` renders an image but exposes it as CONTROL so it can only feed a
#: Control input - so the check below goes through `port_satisfies` rather than demanding equality.
_PORT_FOR: dict[MediaKind, PortKind] = {
    MediaKind.IMAGE: PortKind.IMAGE,
    MediaKind.VIDEO: PortKind.VIDEO,
    MediaKind.AUDIO: PortKind.AUDIO,
}


def _full_registry(tmp_path: Path) -> Registry:
    """The built-ins plus whatever model runners this install can import.

    ``register_models`` is best-effort by design, so on a torch-less machine this is just the
    primitives and loaders. That still covers the invariant; a runtime install covers more.
    """
    registry = build_default_registry()
    register_models(registry, FileTakeStore(tmp_path / "takes"), MemoryPolicy())
    return registry


def test_take_producing_runners_declare_an_output_kind(tmp_path: Path) -> None:
    registry = _full_registry(tmp_path)
    for descriptor in registry.descriptors():
        if not registry.has_runner(descriptor.type):
            continue  # descriptor-only nodes are served and validated but never execute
        produces = registry.runner(descriptor.type).produces_takes
        assert produces == (descriptor.output_kind is not None), (
            f"{descriptor.type}: produces_takes={produces} but output_kind={descriptor.output_kind}"
        )


def test_output_kind_has_a_matching_output_port(tmp_path: Path) -> None:
    registry = _full_registry(tmp_path)
    for descriptor in registry.descriptors():
        if descriptor.output_kind is None:
            continue
        native = _PORT_FOR[descriptor.output_kind]
        kinds = [port.kind for port in descriptor.outputs]
        assert any(port_satisfies(native, kind) for kind in kinds), (
            f"{descriptor.type}: output_kind={descriptor.output_kind} has no output port a "
            f"{native.value} can feed (ports: {[k.value for k in kinds]})"
        )


def test_every_media_kind_is_mapped() -> None:
    """A new MediaKind must be given a port kind here rather than KeyError-ing the suite."""
    assert set(_PORT_FOR) == set(MediaKind)
