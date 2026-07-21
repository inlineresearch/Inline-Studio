"""Two image nodes with no dependencies beyond the host's numpy.

The shape to copy: one `@inline_node`-decorated `NodeRunner` per node. The extension's single
`register(reg)` lives in `__init__.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import ParamField, Port, Widget
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind


@inline_node(
    type="demo/invert",
    title="Invert",
    category="Image",
    icon="wand",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("amount", "Amount", Widget.NUMBER, 1.0, min=0.0, max=1.0, step=0.05),
    ),
)
class Invert(NodeRunner):
    """Blends towards the inverted image by ``amount``."""

    produces_takes = False

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        image = np.asarray(inputs["image"][0], dtype=np.float32)
        amount = float({**Invert.__inline_descriptor__.defaults(), **node.params}["amount"])
        blended = image * (1.0 - amount) + (255.0 - image) * amount
        return NodeResult(outputs={"image": np.clip(blended, 0, 255).astype(np.uint8)})


@inline_node(
    type="demo/brightness",
    title="Brightness",
    category="Image",
    icon="wand",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("scale", "Scale", Widget.NUMBER, 1.2, min=0.0, max=4.0, step=0.1),),
)
class Brightness(NodeRunner):
    produces_takes = False

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        image = np.asarray(inputs["image"][0], dtype=np.float32)
        scale = float({**Brightness.__inline_descriptor__.defaults(), **node.params}["scale"])
        return NodeResult(outputs={"image": np.clip(image * scale, 0, 255).astype(np.uint8)})
