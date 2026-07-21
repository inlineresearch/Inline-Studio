"""A generating node: it produces an image, so it becomes a Frame with take history.

The two things that make a node runnable from the canvas:

* ``output_kind`` on the descriptor - that is what puts the Run control on the node and gives it a
  take strip. A node without it is plumbing that runs when something downstream does.
* ``produces_takes = True`` plus saving through ``ctx.takes`` - that is what the take history and
  the downstream frame reference are built from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import ParamField, Port, Widget
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind


@inline_node(
    type="demo/gradient",
    title="Gradient",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(Port("image", "Tint from (optional)", PortKind.IMAGE),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("width", "Width", Widget.NUMBER, 512, min=64, max=2048, step=64),
        ParamField("height", "Height", Widget.NUMBER, 512, min=64, max=2048, step=64),
        ParamField("hue", "Hue", Widget.NUMBER, 0.6, min=0.0, max=1.0, step=0.05),
    ),
)
class Gradient(NodeRunner):
    produces_takes = True

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        params = {**Gradient.__inline_descriptor__.defaults(), **node.params}
        width, height = int(params["width"]), int(params["height"])
        hue = float(params["hue"])

        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        fade = np.linspace(1.0, 0.3, height, dtype=np.float32)[:, None]
        base = ramp * fade
        image = np.stack(
            [base * hue, base * (1.0 - hue), base * 0.5], axis=-1
        )
        rgb = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

        if ctx.takes is None:
            # No store: the node still emits its image for a downstream node to consume.
            return NodeResult(outputs={"image": rgb})
        take = ctx.takes.save(ctx.run_id, node.id, rgb, params)
        return NodeResult(outputs={"image": rgb}, takes=[take])
