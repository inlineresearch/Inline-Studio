"""LTX-2.5 as three Inline Core nodes: text, image, and first-and-last-frame to video.

One pipeline backs all three. Upstream conditions on a list of ``(image, frame index, strength)``,
so text-to-video is that list empty, image-to-video is one entry at frame 0, and first-and-last is
two entries at frame 0 and the final frame. They are separate nodes rather than one with a mode
switch because their inputs differ, and a node's face should say what it takes.

LTX denoises video and its soundtrack together, so a run produces two takes. Only the video matches
the descriptor's ``output_kind`` and claims the node's canvas slot; the audio is saved beside it and
offered on its own port.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...device.policy import DevicePolicy
from ...errors import ComponentError
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...runtime.store import TakeStore
from .. import pipeline_runtime as rt
from ..video_params import VideoGrid, snap_canvas, snap_frames, video_param_fields

_LABEL = "LTX-2.5"

#: 24 fps and a causal VAE that decodes in blocks of 8 frames plus 1, between 1 and 20 seconds -
#: the window upstream's own duration predictor clamps to. So 25 to 473 frames, and a request for
#: 5 seconds lands on 121, which is upstream's default.
GRID = VideoGrid(fps=24.0, grid=8, offset=1, min_seconds=1.0, max_seconds=20.0)

SHORT_EDGE = 1088
MAX_LONG_EDGE = 1920

#: 64, not 32. The two-stage pipeline runs stage 1 at ``width // 2, height // 2`` and LTX requires
#: both stages to be a multiple of 32, so the size the node asks for has to survive being halved.
#: Upstream asserts this itself (``assert_resolution(..., is_two_stage=True)``).
CANVAS_MULTIPLE = 64

_CANVAS_HINT = "multiple of 64"

#: The fixed schedules the distilled build was trained for: 8 steps in stage 1, 4 in stage 2. They
#: are not a setting, which is why the node shows no step count in fast mode.
DISTILLED_STEPS = 12

MODE_FAST = "fast"
MODE_QUALITY = "quality"


@dataclass(frozen=True)
class Variant:
    node_type: str
    title: str
    first_frame: bool = False
    last_frame: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("lightricks/ltx-2-5-text-to-video", "LTX-2.5 Text to Video"),
    Variant("lightricks/ltx-2-5-image-to-video", "LTX-2.5 Image to Video", first_frame=True),
    Variant(
        "lightricks/ltx-2-5-first-last-frame",
        "LTX-2.5 First and Last Frame",
        first_frame=True,
        last_frame=True,
    ),
)


def _params(variant: Variant) -> tuple[ParamField, ...]:
    fields: list[ParamField] = [
        *video_param_fields(
            GRID,
            short_edge=SHORT_EDGE,
            multiple=CANVAS_MULTIPLE,
            max_long_edge=MAX_LONG_EDGE,
            canvas_hint=_CANVAS_HINT,
        ),
        ParamField(
            "mode", "Mode", Widget.SELECT, MODE_FAST,
            options=(
                Option(MODE_FAST, "Fast (distilled, 12 steps)"),
                Option(MODE_QUALITY, "Quality (dev, guided)"),
            ),
        ),
        ParamField("generate_audio", "Generate audio", Widget.BOOLEAN, True),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
        # Quality mode only. Fast mode runs a fixed sigma schedule, so a step count there would be
        # a control that does nothing.
        ParamField(
            "num_inference_steps", "Steps (quality mode)", Widget.NUMBER, 30,
            min=1, max=200, step=1, advanced=True,
        ),
        ParamField(
            "guidance", "Guidance (quality mode)", Widget.NUMBER, 3.0,
            min=1.0, max=15.0, step=0.5, advanced=True,
        ),
        ParamField("enhance_prompt", "Expand the prompt", Widget.BOOLEAN, False, advanced=True),
        ParamField(
            "model", "Diffusion model", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        ),
        ParamField(
            "text_encoder", "Text encoder", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        ),
        ParamField("vae", "Video VAE", Widget.SELECT, "", options_from="vae", advanced=True),
        ParamField(
            "upscaler", "Spatial upscaler", Widget.SELECT, "",
            options_from="latent_upscale_models", advanced=True,
        ),
    ]
    return tuple(fields)


def _inputs(variant: Variant) -> tuple[Port, ...]:
    ports = [
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "Video VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
        Port("lora", "LoRA", PortKind.LORA, required=False),
    ]
    if variant.first_frame:
        ports.append(Port("image", "First frame", PortKind.IMAGE, required=False))
    if variant.last_frame:
        ports.append(Port("last_image", "Last frame", PortKind.IMAGE, required=False))
    # An IC-LoRA (Control LoRA) learns a transform from a reference clip, so it needs one at
    # generation time too. Present on every variant: the reference is orthogonal to keyframes.
    ports.append(
        Port("reference", "Reference clip (Control LoRA)", PortKind.VIDEO, required=False)
    )
    return tuple(ports)


def descriptor(variant: Variant) -> NodeDescriptor:
    return NodeDescriptor(
        type=variant.node_type,
        title=variant.title,
        category="Generate",
        icon="wand",
        output_kind=MediaKind.VIDEO,
        inputs=_inputs(variant),
        outputs=(Port("video", "Video", PortKind.VIDEO), Port("audio", "Audio", PortKind.AUDIO)),
        params=_params(variant),
    )


DESCRIPTORS: dict[str, NodeDescriptor] = {v.node_type: descriptor(v) for v in VARIANTS}


# --- the call ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Conditioning:
    """One wired keyframe, as the frame index it pins and the image behind it."""

    value: Any
    frame_index: int
    strength: float = 1.0


@dataclass(frozen=True)
class Request:
    """Everything the pipeline needs, resolved from params and wired inputs.

    Split out from the runner so the awkward parts - snapping a duration onto the frame grid,
    rounding a canvas that has to survive being halved - are unit-testable with no weights and no
    GPU.
    """

    prompt: str
    num_frames: int
    width: int
    height: int
    num_inference_steps: int
    guidance: float
    seed: int
    mode: str
    generate_audio: bool
    enhance_prompt: bool
    conditionings: tuple[Conditioning, ...] = ()
    #: A reference clip for a Control LoRA, as a local path. Upstream reads the file itself rather
    #: than taking pixels, so this stays a path all the way down.
    reference: str | None = None

    @property
    def seconds(self) -> float:
        return GRID.seconds(self.num_frames)

    @property
    def build(self) -> str:
        """Which transformer this mode loads."""
        return "dev" if self.mode == MODE_QUALITY else "distilled"

    @property
    def stage_1_size(self) -> tuple[int, int]:
        """The half canvas stage 1 renders at, which must still be a multiple of 32."""
        return self.width // 2, self.height // 2


def build_request(
    variant: Variant, params: dict[str, Any], inputs: dict[str, list[Any]]
) -> Request:
    """Resolve a node's params and wiring into a request the pipeline will accept."""
    prompt = rt.first_str(inputs.get("prompt"))
    if not prompt:
        raise ComponentError(f"{variant.title} needs a prompt.")
    frames = snap_frames(float(params.get("duration") or GRID.min_seconds), GRID)
    width, height = snap_canvas(
        int(params.get("width") or SHORT_EDGE),
        int(params.get("height") or SHORT_EDGE),
        multiple=CANVAS_MULTIPLE,
        minimum=CANVAS_MULTIPLE,
    )
    mode = str(params.get("mode") or MODE_FAST)

    conditionings: list[Conditioning] = []
    if variant.first_frame and (first := rt.first(inputs.get("image"))) is not None:
        conditionings.append(Conditioning(first, 0))
    if variant.last_frame and (last := rt.first(inputs.get("last_image"))) is not None:
        conditionings.append(Conditioning(last, frames - 1))
    if variant.first_frame and not conditionings:
        raise ComponentError(f"{variant.title} needs an image wired to its First frame input.")

    reference = rt.first(inputs.get("reference"))
    return Request(
        prompt=prompt,
        num_frames=frames,
        width=width,
        height=height,
        num_inference_steps=(
            max(1, int(params.get("num_inference_steps") or 30))
            if mode == MODE_QUALITY
            else DISTILLED_STEPS
        ),
        guidance=float(params.get("guidance") or 3.0),
        seed=rt.resolve_seed(params.get("seed")),
        mode=mode,
        generate_audio=bool(params.get("generate_audio", True)),
        enhance_prompt=bool(params.get("enhance_prompt", False)),
        reference=rt.media_path(reference, "Reference clip") if reference else None,
        conditionings=tuple(conditionings),
    )


def _conditioning_path(value: Any, label: str, scratch: Path) -> str:
    """A file path for a wired keyframe.

    LTX's image conditioning takes a path rather than a decoded image, because it re-compresses the
    frame at the CRF the checkpoint was trained on before encoding it. Handing it an already-decoded
    array would skip that and condition on something the model never saw in training, so a wired
    image that only exists in memory is written out rather than passed through.
    """
    direct = getattr(value, "path", None) or getattr(value, "uri", None)
    if isinstance(direct, str | Path) and Path(str(direct)).is_file():
        return str(direct)
    image = rt.load_image(value, label)
    out = scratch / f"keyframe-{abs(id(value))}.png"
    image.save(out)
    return str(out)


def call_kwargs(request: Request, scratch: Path, label: str = _LABEL) -> dict[str, Any]:
    """The keyword arguments handed to the pipeline.

    ``num_frames`` is passed explicitly and is already on the ``8n + 1`` grid, so upstream's own
    snap is a no-op. That matters: upstream rounds **down** and our params round **up**, and letting
    both run would quietly render fewer frames than the node's duration field promised.
    """
    from .vendor.ltx_pipelines.utils.args import ImageConditioningInput

    args: dict[str, Any] = {
        "prompt": request.prompt,
        "seed": request.seed,
        "height": request.height,
        "width": request.width,
        "frame_rate": GRID.fps,
        "num_frames": request.num_frames,
        "enhance_prompt": request.enhance_prompt,
        "images": [
            ImageConditioningInput(
                path=_conditioning_path(c.value, label, scratch),
                frame_idx=c.frame_index,
                strength=c.strength,
            )
            for c in request.conditionings
        ],
    }
    if request.reference:
        # Only the IC-LoRA pipeline takes this, and it is the pipeline `load_pipeline` picks when a
        # reference is wired. Strength 1.0 keeps the reference tokens clean, which is how the
        # adapter saw them in training.
        args["video_conditioning"] = [(request.reference, 1.0)]
    return args


# --- the runner ----------------------------------------------------------------------------------


def register_ltx25(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register all three nodes. Called best-effort by server.bootstrap."""
    for variant in VARIANTS:
        registry.register(DESCRIPTORS[variant.node_type], Ltx25Runner(store, policy, variant))


class Ltx25Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy, variant: Variant) -> None:
        self._store = store
        self._policy = policy
        self._variant = variant

    def run(
        self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext
    ) -> NodeResult:
        # Deferred: the vendored package pulls the whole LTX stack, which a torch-less install
        # cannot import and does not need in order to serve this descriptor.
        from .pipeline import load_pipeline, render

        params = {**DESCRIPTORS[self._variant.node_type].defaults(), **node.params}
        request = build_request(self._variant, params, inputs)
        rt.raise_if_cancelled(ctx)

        ctx.emitter.emit(
            rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…")
        )
        pipe = load_pipeline(
            self._policy,
            params=params,
            request=request,
            transformer=rt.component_ref(inputs, "model", "diffusion", _LABEL),
            video_vae=rt.component_ref(inputs, "vae", "vae", _LABEL),
            text_encoder=rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL),
            loras=rt.lora_stack(inputs, _LABEL),
        )

        with tempfile.TemporaryDirectory(prefix="ltx25-") as scratch:
            call = call_kwargs(request, Path(scratch), self._variant.title)

            def on_step(done: int, total: int) -> None:
                rt.raise_if_cancelled(ctx)
                ctx.emitter.emit(
                    rt.progress_event(
                        ctx, node, Phase.SAMPLE,
                        done / total if total else 0.0,
                        status=f"Step {done} of {total}",
                    )
                )

            ctx.emitter.emit(
                rt.progress_event(ctx, node, Phase.SAMPLE, 0.0, status="Generating…")
            )
            result = render(
                pipe,
                request,
                call,
                on_step=on_step,
                cancel_check=lambda: rt.raise_if_cancelled(ctx),
            )

        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status="Saving…"))
        return self._result(node, ctx, request, result)

    def _result(
        self, node: Node, ctx: ExecutionContext, request: Request, result: Any
    ) -> NodeResult:
        meta = {
            "model": f"LTX-2.5 {request.build}",
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "duration_seconds": round(request.seconds, 3),
            "fps": GRID.fps,
            "steps": request.num_inference_steps,
            "seed": request.seed,
        }
        video_take = self._store.save_video(
            ctx.run_id, node.id, result.frames, meta,
            fps=GRID.fps, audio=result.waveform, sample_rate=result.sample_rate,
        )
        takes = [video_take]
        outputs: dict[str, Any] = {"video": video_take}
        if result.waveform is not None and result.sample_rate:
            audio_take = self._store.save_audio(
                ctx.run_id, node.id, result.waveform, meta, sample_rate=int(result.sample_rate)
            )
            takes.append(audio_take)
            outputs["audio"] = audio_take
        return NodeResult(outputs=outputs, takes=takes)
