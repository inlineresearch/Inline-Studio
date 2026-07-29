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
    # A control map, not a plain image: kind CONTROL so it only feeds a gen node's Control input,
    # never the img2img Image input (that would run image-to-image from the black map, not control).
    outputs=(Port("image", "Control map", PortKind.CONTROL),),
    params=(
        ParamField(
            "type", "Type", Widget.SELECT, "pose",
            options=(
                Option("pose", "OpenPose (pose)"),
                Option("depth_anything", "Depth-Anything V2 (depth, for Krea 2)"),
                Option("depth", "MiDaS (depth)"),
                Option("canny", "Canny edges (no model)"),
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
    elif kind == "depth_anything":
        det = _DepthAnythingDetector()
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


class _DepthAnythingDetector:
    """Depth-Anything-V2-Large depth, the estimator the Krea 2 depth control-LoRA trained on.
    Returns a grayscale RGB depth map (near = white), per-image normalized. Auto-fetches into the HF
    cache on first use, the same fetch-once posture as the controlnet_aux detectors."""

    _MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

    _processor: Any
    _model: Any

    def __init__(self) -> None:
        try:
            import transformers
        except ImportError as error:
            raise ComponentError(
                "Depth-Anything needs transformers (ships in the runtime extra). Reinstall it: "
                "uv pip install -e '.[runtime]'."
            ) from error
        tf: Any = transformers
        self._processor = tf.AutoImageProcessor.from_pretrained(self._MODEL_ID)
        self._model = tf.AutoModelForDepthEstimation.from_pretrained(self._MODEL_ID).eval()

    def __call__(
        self, image: Any, detect_resolution: int = 512, image_resolution: int = 512
    ) -> Any:
        # Depth-Anything's processor governs its own input size, so the resolution knobs (meant for
        # the controlnet_aux detectors) are accepted for a uniform call but not used here.
        del detect_resolution, image_resolution
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        with torch.no_grad():
            inputs = self._processor(images=[image], return_tensors="pt")
            depth: Any = self._model(**inputs).predicted_depth[None].float()
            depth = F.interpolate(
                depth, size=(image.height, image.width), mode="bilinear", align_corners=False
            )[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        array = (depth.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")


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
