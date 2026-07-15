"""Shared test fixtures: a fake model node, a registry, and graph/context builders."""

from __future__ import annotations

from typing import Any

from inline_core.device.auto import AutoDevicePolicy
from inline_core.graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from inline_core.graph.registry import Registry, build_default_registry
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import Graph, Node, PortKind, parse_graph
from inline_core.media import MediaKind
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import Phase, ProgressEmitter, ProgressEvent
from inline_core.takes import Take

FAKE_MODEL = NodeDescriptor(
    type="fake/model",
    title="Fake",
    category="Image",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        Port("image", "Init image", PortKind.IMAGE, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("steps", "Steps", Widget.NUMBER, 8),
        ParamField("seed", "Seed", Widget.SEED, -1, advanced=True),
    ),
)


class FakeModelRunner(NodeRunner):
    produces_takes = True

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        ctx.emitter.emit(
            ProgressEvent(ctx.run_id, node.id, Phase.SAMPLE, 0.5, step=1, step_count=2)
        )
        take = Take(
            id=f"take-{node.id}",
            run_id=ctx.run_id,
            node_id=node.id,
            kind=MediaKind.IMAGE,
            uri=f"mem://{node.id}",
            hash=f"h-{node.id}",
            params=dict(node.params),
        )
        return NodeResult(outputs={"image": take}, takes=[take])


def make_registry() -> Registry:
    registry = build_default_registry()
    registry.register(FAKE_MODEL, FakeModelRunner())
    return registry


def build_graph(model_params: dict[str, Any] | None = None) -> Graph:
    return parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "p1", "type": "input/text", "params": {"text": "a fox"}},
                {
                    "id": "m1",
                    "type": "fake/model",
                    "params": model_params or {},
                    "inputs": {"prompt": {"from": "p1", "output": "text"}},
                },
            ],
        }
    )


def build_ctx(emitter: ProgressEmitter, run_id: str = "run1") -> ExecutionContext:
    return ExecutionContext(
        run_id=run_id, policy=AutoDevicePolicy(), emitter=emitter, cancel=CancelToken()
    )
