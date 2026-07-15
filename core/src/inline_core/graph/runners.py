"""Node runners: the behavior half of a node. Source runners are pure; model runners lower to
components (see models/runner.py). Every runner returns its output values and any takes it produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..errors import ComponentError
from ..runtime.context import ExecutionContext
from ..takes import AssetRef, Take
from .descriptor import NodeDescriptor, ParamField, Port, Widget
from .schema import Node, PortKind


@dataclass
class NodeResult:
    outputs: dict[str, Any]
    takes: list[Take] = field(default_factory=list)


class NodeRunner(ABC):
    produces_takes: ClassVar[bool] = False

    @abstractmethod
    def run(
        self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext
    ) -> NodeResult: ...


class TextInputRunner(NodeRunner):
    produces_takes = False

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        return NodeResult(outputs={"text": str(node.params.get("text", ""))})


class ImageInputRunner(NodeRunner):
    produces_takes = False

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        return NodeResult(outputs={"image": _asset_ref(node.params.get("asset"))})


def _asset_ref(raw: Any) -> AssetRef:
    if isinstance(raw, dict):
        ref = raw.get("ref")
        if ref == "asset":
            return AssetRef(ref="asset", id=str(raw.get("id", "")))
        if ref == "path":
            return AssetRef(ref="path", path=str(raw.get("path", "")))
    raise ComponentError("An image input node needs a valid asset reference.")


TEXT_INPUT = NodeDescriptor(
    type="input/text",
    title="Prompt",
    category="Input",
    outputs=(Port("text", "Text", PortKind.TEXT),),
    params=(ParamField("text", "Text", Widget.TEXTAREA, ""),),
    icon="type",
)

IMAGE_INPUT = NodeDescriptor(
    type="input/image",
    title="Image",
    category="Input",
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    icon="image",
)
