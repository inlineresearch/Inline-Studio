"""The four MiniMax H3 nodes: text, image, first-and-last-frame, and omni-reference to video.

Four separate nodes rather than one with a mode switch, because the inputs differ and a node's face
should say what it takes. They share one loader and one run path; only the partition and the call
arguments differ.

H3 generates video and its stereo soundtrack in a single denoising pass, so a take is one MP4 with
the audio muxed in. The soundtrack is also exposed on its own port for wiring downstream - see
``_result`` for why both takes are returned but only the video claims the node's card.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...device.policy import DevicePolicy
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...runtime.store import TakeStore
from .. import pipeline_runtime as rt
from ..references import ReferenceKind, ReferenceLimits, collect_references, describe
from ..video_params import VideoGrid, snap_canvas, snap_frames, video_param_fields

logger = logging.getLogger("inline_core.minimaxh3")

#: Names this node in the error a mis-wired handle raises.
_LABEL = "MiniMax H3"

#: 24 fps, decodable in blocks of 17 frames plus 5, between 5 and 15 seconds: 124 to 345 frames.
GRID = VideoGrid(fps=24.0, grid=17, offset=5, min_seconds=5.0, max_seconds=15.0)

#: A 768 pixel short edge is H3's trained canvas; 1344 is the long edge it was trained at.
SHORT_EDGE = 768
MAX_LONG_EDGE = 1344
CANVAS_MULTIPLE = 32

#: Up to 9 images, 3 video clips and 3 audio clips, 12 in total.
REFERENCE_LIMITS = ReferenceLimits(max_images=9, max_videos=3, max_audio=3, max_total=12)

_CANVAS_HINT = "multiples of 32; 960x544 renders about 2.3x faster per step than 1344x768"


@dataclass(frozen=True)
class Variant:
    """One node: which partition it needs and which inputs it offers."""

    node_type: str
    title: str
    partition: str  # "fl2va" | "ref2va"
    first_frame: bool = False
    last_frame: bool = False
    references: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("minimax/h3-text-to-video", "MiniMax H3 Text to Video", "fl2va"),
    Variant(
        "minimax/h3-image-to-video", "MiniMax H3 Image to Video", "fl2va", first_frame=True
    ),
    Variant(
        "minimax/h3-first-last-frame",
        "MiniMax H3 First and Last Frame",
        "fl2va",
        first_frame=True,
        last_frame=True,
    ),
    Variant(
        "minimax/h3-reference-to-video",
        "MiniMax H3 Reference to Video",
        "ref2va",
        references=True,
    ),
)


def _params(variant: Variant) -> tuple[ParamField, ...]:
    """Duration, canvas, steps and seed. **No guidance and no negative prompt**: the released
    checkpoints are guidance-distilled, so neither exists rather than being quietly ignored."""
    fields: list[ParamField] = [
        *video_param_fields(
            GRID,
            short_edge=SHORT_EDGE,
            multiple=CANVAS_MULTIPLE,
            max_long_edge=MAX_LONG_EDGE,
            canvas_hint=_CANVAS_HINT,
        ),
        # Counts sigma grid points, terminal zero included, so it drives one model evaluation less.
        ParamField("num_inference_steps", "Steps", Widget.NUMBER, 50, min=1, max=200, step=1),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
    ]
    if variant.references:
        fields.append(
            ParamField(
                "ref_image_size", "Reference detail", Widget.SELECT, "match",
                options_from=None, advanced=True,
            )
        )
    fields.append(
        ParamField(
            "model", "Diffusion model", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        )
    )
    fields.append(
        ParamField(
            "text_encoder", "Text encoder", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        )
    )
    fields.append(
        ParamField("vae", "Video VAE", Widget.SELECT, "", options_from="vae", advanced=True)
    )
    return tuple(fields)


def _inputs(variant: Variant) -> tuple[Port, ...]:
    ports = [
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # The same component handles every other Core node carries: wire a load/* subnode to
        # override the dropdown.
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "Video VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
        # Adapters fuse into each block as it streams, before factorisation and quantisation. A
        # LoRA trained on either partition loads on both: they are the same architecture.
        Port("lora", "LoRA", PortKind.LORA, required=False),
    ]
    if variant.first_frame:
        ports.append(Port("image", "First frame", PortKind.IMAGE, required=False))
    if variant.last_frame:
        ports.append(Port("last_image", "Last frame", PortKind.IMAGE, required=False))
    if variant.references:
        # A list port: wiring order is the numbering the prompt addresses, so it must be preserved.
        ports.append(Port("references", "References", PortKind.IMAGE_LIST, required=False))
        ports.append(Port("video", "Reference video", PortKind.VIDEO, required=False))
        ports.append(Port("audio", "Reference audio", PortKind.AUDIO, required=False))
    return tuple(ports)


def descriptor(variant: Variant) -> NodeDescriptor:
    return NodeDescriptor(
        type=variant.node_type,
        title=variant.title,
        category="Generate",
        icon="wand",
        output_kind=MediaKind.VIDEO,
        inputs=_inputs(variant),
        outputs=(
            Port("video", "Video", PortKind.VIDEO),
            Port("audio", "Audio", PortKind.AUDIO),
        ),
        params=_params(variant),
    )


DESCRIPTORS: dict[str, NodeDescriptor] = {v.node_type: descriptor(v) for v in VARIANTS}


# --- the call ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """Everything the blocks need, resolved from params and wired inputs.

    Split out from the runner so the awkward parts - snapping a duration onto the frame grid,
    rounding a canvas, ordering references - are unit-testable with no weights and no GPU.
    """

    prompt: str
    num_frames: int
    width: int
    height: int
    num_inference_steps: int
    seed: int
    partition: str
    references: tuple[Any, ...] = ()

    @property
    def seconds(self) -> float:
        return GRID.seconds(self.num_frames)


def build_request(
    variant: Variant, params: dict[str, Any], inputs: dict[str, list[Any]]
) -> Request:
    """Resolve a node's params and wiring into a request the blocks will accept."""
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
    references: tuple[Any, ...] = ()
    if variant.references:
        references = collect_references(inputs, limits=REFERENCE_LIMITS)
        if not references:
            raise ComponentError(
                f"{variant.title} needs at least one reference wired to its References, "
                "Reference video or Reference audio input."
            )
    return Request(
        prompt=prompt,
        num_frames=frames,
        width=width,
        height=height,
        num_inference_steps=max(1, int(params.get("num_inference_steps") or 50)),
        seed=rt.resolve_seed(params.get("seed")),
        partition=variant.partition,
        references=references,
    )


def _h3_reference(ref: Any, label: str) -> Any:
    """Adapt one of our `Reference`s into the vendored `MiniMaxH3Reference` the blocks require.

    `models/references.py` is the shared, model-agnostic seam - ordering and numbering are the same
    problem for any model that reads several media - so the H3-shaped dataclass is built here at the
    boundary rather than leaking a vendored type into it. Order is preserved because order *is* the
    numbering the prompt addresses.
    """
    from ..references import ReferenceKind
    from .vendor import MiniMaxH3Reference

    if ref.kind is ReferenceKind.IMAGE:
        return MiniMaxH3Reference(image=rt.load_image(ref.value, label))
    if ref.kind is ReferenceKind.VIDEO:
        return MiniMaxH3Reference(video=_media_path(ref.value, label, "video"))
    return MiniMaxH3Reference(audio=_media_path(ref.value, label, "audio"))


def _media_path(value: Any, label: str, kind: str) -> str:
    """A path the blocks can open. They decode video and audio themselves, so a path is what they
    want; anything that is not one is a wiring error worth naming."""
    path = getattr(value, "path", None) or getattr(value, "uri", None) or value
    if isinstance(path, str | Path) and Path(str(path)).is_file():
        return str(path)
    raise ComponentError(f"{label} could not read the wired {kind} reference.")


def call_kwargs(request: Request, variant: Variant, inputs: dict[str, list[Any]]) -> dict[str, Any]:
    """The keyword arguments handed to the blocks.

    Note there is exactly one ``generator``: H3 draws the conditioning, video and audio noise from
    it in sequence, so changing the duration also changes the soundtrack. Decoupling them means
    supplying ``latents`` and ``audio_latents`` directly, whose shapes are internal to the blocks,
    so that is deliberately left until it can be checked against a real run rather than guessed.
    """
    call: dict[str, Any] = {
        "prompt": request.prompt,
        "num_frames": request.num_frames,
        "height": request.height,
        "width": request.width,
        "num_inference_steps": request.num_inference_steps,
        "output_type": "pil",
    }
    if variant.references:
        call["references"] = [_h3_reference(ref, variant.title) for ref in request.references]
    else:
        first = rt.first(inputs.get("image"))
        last = rt.first(inputs.get("last_image"))
        if first is not None:
            call["image"] = rt.load_image(first, variant.title)
        if last is not None:
            call["last_image"] = rt.load_image(last, variant.title)
    return call


# --- the runner ----------------------------------------------------------------------------------


def register_minimax_h3(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register all four nodes. Called best-effort by server.bootstrap."""
    for variant in VARIANTS:
        registry.register(DESCRIPTORS[variant.node_type], MiniMaxH3Runner(store, policy, variant))


class MiniMaxH3Runner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy, variant: Variant) -> None:
        self._store = store
        self._policy = policy
        self._variant = variant

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        from .pipeline import load_pipeline, render_staged  # deferred: vendored blocks

        params = {**DESCRIPTORS[self._variant.node_type].defaults(), **node.params}
        request = build_request(self._variant, params, inputs)
        logger.info(
            "MiniMax H3 %s: %dx%d, %d frames (%.2fs), %d steps, refs=%s | %s",
            self._variant.partition, request.width, request.height, request.num_frames,
            request.seconds, request.num_inference_steps, describe(request.references),
            rt.device_report(self._policy),
        )

        rt.raise_if_cancelled(ctx)
        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        # Wired component handles from load/* subnodes override the dropdowns, which in turn
        # override the default filenames, as on the image nodes. All of it reaches the cache key
        # through the resolved paths themselves.
        pipe = load_pipeline(
            self._policy, params=params, partition=request.partition,
            transformer=rt.component_ref(inputs, "model", "diffusion", _LABEL),
            video_vae=rt.component_ref(inputs, "vae", "vae", _LABEL),
            text_encoder=rt.component_ref(inputs, "text_encoder", "text_encoder", _LABEL),
            loras=rt.lora_stack(inputs, _LABEL),
        )

        call = call_kwargs(request, self._variant, inputs)
        call["generator"] = torch.Generator(device="cpu").manual_seed(request.seed)
        started = time.perf_counter()
        try:
            state = render_staged(pipe, self._policy.placement('denoiser').device, **call)
        except CancelledError:
            rt.free_vram()
            raise
        except torch.cuda.OutOfMemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(request)) from error
        except MemoryError as error:
            rt.free_vram()
            raise ComponentError(_oom(request, host=True)) from error
        logger.info("MiniMax H3 sampled in %.1fs", time.perf_counter() - started)
        rt.free_vram()

        ctx.emitter.emit(rt.progress_event(ctx, node, Phase.SAVE, 1.0, status="Saving…"))
        return self._result(node, ctx, state, request)

    def _result(
        self, node: Node, ctx: ExecutionContext, state: Any, request: Request
    ) -> NodeResult:
        """One muxed MP4 plus the bare soundtrack.

        Both takes are returned so both are persisted into the project, but only the video's kind
        matches this node's ``output_kind``, so only it claims the card. See
        ``studio.generation._save_take``.
        """
        videos = state.get("videos")
        audio = state.get("audio")
        sample_rate = state.get("sampling_rate")
        if videos is None or not len(videos):
            raise ComponentError("MiniMax H3 returned no video.")
        frames = videos[0]
        # `if audio` is ambiguous when the blocks hand back a tensor rather than a list, so
        # the emptiness check is explicit.
        waveform = audio[0] if audio is not None and len(audio) else None
        meta = {
            "model": f"minimax-h3-{request.partition}",
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
            ctx.run_id, node.id, frames, meta,
            fps=GRID.fps, audio=waveform, sample_rate=sample_rate,
        )
        takes = [video_take]
        outputs: dict[str, Any] = {"video": video_take}
        if waveform is not None and sample_rate:
            audio_take = self._store.save_audio(
                ctx.run_id, node.id, waveform, meta, sample_rate=int(sample_rate)
            )
            takes.append(audio_take)
            outputs["audio"] = audio_take
        return NodeResult(outputs=outputs, takes=takes)


def _oom(request: Request, *, host: bool = False) -> str:
    where = "System RAM" if host else "VRAM"
    return (
        f"{where} ran out at {request.width}x{request.height} for {request.seconds:.1f}s. "
        "Canvas is the biggest lever: 960x544 needs far less than 1344x768 and renders about "
        "2.3x faster per step. A shorter duration helps too."
    )


__all__ = [
    "DESCRIPTORS",
    "GRID",
    "REFERENCE_LIMITS",
    "ReferenceKind",
    "Request",
    "VARIANTS",
    "Variant",
    "build_request",
    "call_kwargs",
    "register_minimax_h3",
]
