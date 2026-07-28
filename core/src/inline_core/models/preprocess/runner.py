"""Apply ControlNet: turn an input image into a control map (pose / depth / canny) for a ControlNet.

A single preprocessing node, ``control/apply``. controlnet_aux ships in the ``runtime`` extra but is
imported inside the detector build on purpose - kept out of module top so the node's descriptor is
served even on a runtime-less install; running it then fails with a clear "install the runtime
extra" error rather than the node silently vanishing. The detector weights download once into the HF
cache on first use (opt-in, the same fetch-once posture the auto-captioner uses)."""

from __future__ import annotations

import logging
from typing import Any

from ...config import models_dir
from ...errors import ComponentError
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...runtime.store import TakeStore
from .. import pipeline_runtime as rt

logger = logging.getLogger("inline_core.preprocess")

#: controlnet_aux's annotator weights (OpenPose, MiDaS depth, ...). Canny needs no weights.
_ANNOTATOR_REPO = "lllyasviel/Annotators"
_LABEL = "Apply ControlNet"

CONTROL_APPLY = NodeDescriptor(
    type="control/apply",
    title="Apply ControlNet",
    category="Control",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Control map", PortKind.IMAGE),),
    params=(
        ParamField(
            "type", "Type", Widget.SELECT, "pose",
            options=(
                Option("pose", "OpenPose (pose)"),
                Option("depth", "Depth"),
                Option("canny", "Canny (edges)"),
            ),
        ),
        ParamField(
            "detect_resolution", "Detail resolution", Widget.NUMBER, 512,
            min=256, max=1536, step=64, advanced=True,
        ),
    ),
)


def register_control_apply(registry: Any, store: TakeStore, policy: Any) -> None:
    """Register the Apply ControlNet node. Called best-effort by server.bootstrap. ``policy`` is
    accepted for a uniform signature but unused - preprocessing runs on CPU."""
    del policy
    registry.register(CONTROL_APPLY, ControlApplyRunner(store))


# Detectors are heavy to build (weights load / download), so build once and reuse across runs.
_DETECTORS: dict[str, Any] = {}


def _annotator_source(*files: str) -> str:
    # Prefer weights the user prefetched into models/annotators/ (controlnet_aux loads them flat
    # from a local dir); else the HF repo, which auto-fetches once into the HF cache. All files must
    # be present for the local dir, or from_pretrained would fail on the missing one.
    local = models_dir() / "annotators"
    if all((local / f).is_file() for f in files):
        return str(local)
    return _ANNOTATOR_REPO


def _detector(kind: str) -> Any:
    cached = _DETECTORS.get(kind)
    if cached is not None:
        return cached
    try:
        from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector
    except ImportError as error:
        raise ComponentError(
            "Apply ControlNet needs controlnet_aux (ships in the runtime extra). Reinstall it: "
            "uv pip install -e '.[runtime]'."
        ) from error
    if kind == "canny":
        det: Any = CannyDetector()
    elif kind == "depth":
        det = MidasDetector.from_pretrained(_annotator_source("dpt_hybrid-midas-501f0c75.pt"))
    elif kind == "pose":
        det = OpenposeDetector.from_pretrained(
            _annotator_source("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth")
        )
    else:
        raise ComponentError(f"Unknown control type {kind!r}. Use pose, depth or canny.")
    _DETECTORS[kind] = det
    return det


class ControlApplyRunner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore) -> None:
        self._store = store

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        image_ref = rt.first(inputs.get("image"))
        if image_ref is None:
            raise ComponentError("Apply ControlNet needs an input image.")
        image = rt.load_image(image_ref, _LABEL)
        params = {**CONTROL_APPLY.defaults(), **node.params}
        kind = str(params.get("type") or "pose")
        res = int(params.get("detect_resolution") or 512)

        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status=f"{kind} map…"))
        rt.raise_if_cancelled(ctx)
        detector = _detector(kind)
        logger.info("Apply ControlNet: %s map at %dpx", kind, res)
        control = detector(image, detect_resolution=res, image_resolution=res)

        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status="Saving…"))
        take = self._store.save(ctx.run_id, node.id, control, {"type": kind})
        return NodeResult(outputs={"image": take}, takes=[take])
