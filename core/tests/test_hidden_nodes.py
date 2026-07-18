"""Source inputs and the not-yet-runnable primitives (encode/sample/decode) are served but marked
hidden, so the UI never offers them - generation stays one-click on high-level model nodes. The
``load/*`` loaders are the exception: they have runners and are visible (they feed a model node)."""

from __future__ import annotations

from inline_core.graph.registry import build_default_registry
from inline_core.server.serialize import descriptor_json

_HIDDEN = {
    "input/text",
    "input/image",
    "encode/text",
    "latent/empty",
    "sample",
    "vae/decode",
    "vae/encode",
}

_VISIBLE = {"load/diffusion-model", "load/vae", "load/text-encoder"}


def test_primitives_and_source_nodes_are_hidden() -> None:
    registry = build_default_registry()
    for node_type in _HIDDEN:
        assert registry.get(node_type).hidden is True, node_type


def test_loaders_are_visible_and_runnable() -> None:
    registry = build_default_registry()
    for node_type in _VISIBLE:
        assert registry.get(node_type).hidden is False, node_type
        assert registry.runner(node_type) is not None, node_type


def test_descriptor_json_emits_hidden_only_when_true() -> None:
    registry = build_default_registry()
    # A hidden node serializes with hidden: true...
    assert descriptor_json(registry.get("sample"))["hidden"] is True
    # ...while a normal (non-hidden) descriptor omits the key entirely.
    from inline_core.graph.descriptor import NodeDescriptor

    visible = descriptor_json(NodeDescriptor(type="x/model", title="X", category="Generate"))
    assert "hidden" not in visible
