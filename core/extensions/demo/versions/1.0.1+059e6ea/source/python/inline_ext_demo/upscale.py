"""A node that needs a weight file.

The weights are declared as data in ``inline-extension.json``, not fetched here - Inline Studio's
existing model popup handles the download into ``models/upscale_models/``. Load with
``local_files_only``-style behaviour and report a clear error when the file is absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from inline_core.config import models_dir
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import ParamField, Port, Widget
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind

WEIGHTS = "RealESRGAN_x4.pth"


@inline_node(
    type="demo/upscale",
    title="Demo Upscale",
    category="Enhance",
    icon="wand",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField(
            "model",
            "Weights",
            Widget.SELECT,
            WEIGHTS,
            # Filled from whatever is installed under models/upscale_models/.
            options_from="upscale_models",
        ),
        ParamField("scale", "Scale", Widget.NUMBER, 2, min=2, max=4, step=2),
    ),
)
class Upscale(NodeRunner):
    produces_takes = False

    def run(self, node: Any, inputs: dict[str, list[Any]], ctx: Any) -> NodeResult:
        params = {**Upscale.__inline_descriptor__.defaults(), **node.params}
        weights = models_dir() / "upscale_models" / str(params["model"] or WEIGHTS)
        if not weights.is_file():
            raise FileNotFoundError(
                f"{weights.name} is not installed. Open the node's model popup to download it."
            )
        # A real upscaler would load `weights` here; nearest-neighbour keeps the demo dependency-free.
        image = np.asarray(inputs["image"][0])
        factor = int(params["scale"])
        return NodeResult(outputs={"image": np.repeat(np.repeat(image, factor, 0), factor, 1)})
