"""Z-Image Turbo runner: prompt (+ optional image) -> one rendered take.

A single generation node, ``alibaba/z-image-turbo``, backed by diffusers' ``ZImagePipeline``
(text-to-image) and ``ZImageImg2ImgPipeline`` (when an image is wired in). The heavy pipeline is
built once and cached across runs; only the descriptor is cheap. Placement (device, dtype, offload,
tiling) comes from the DevicePolicy — the runner never picks a device itself. Decoded images are
handed to the TakeStore, which owns bytes/hash/uri.

torch + diffusers are imported at module top on purpose: an absent ``zimage`` extra makes this
import raise, and ``server.bootstrap`` skips the model (best-effort) so the engine still boots.
"""

from __future__ import annotations

import os
import random
from threading import Lock
from typing import Any

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    ZImageImg2ImgPipeline,
    ZImagePipeline,
    ZImageTransformer2DModel,
)

from ...device.policy import DevicePolicy, Placement, Profile
from ...device.types import DeviceKind, DType
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase, ProgressEvent
from ...runtime.store import TakeStore
from ...takes import AssetRef
from . import requirements as reqs

# The models this node needs — the diffusion transformer plus a VAE, text-encoder, tokenizer and
# scheduler — are assembled entirely from files under models/ (see `requirements.py`). **Nothing is
# ever downloaded here.** Every diffusers/transformers load below runs with local_files_only=True,
# so a missing model is a clear error pointing at the node's model popup — never a silent fetch from
# Hugging Face. Models arrive by exactly two paths: the user drops files under models/, or the popup
# downloads them into models/.
_SEED_MAX = 2**31 - 1


ZIMAGE = NodeDescriptor(
    type="alibaba/z-image-turbo",
    title="Z-Image Turbo",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # Optional: wire an image to run img2img instead of text-to-image.
        Port("image", "Image (img2img)", PortKind.IMAGE, required=False),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(
        ParamField("negative_prompt", "Negative prompt", Widget.TEXTAREA, ""),
        ParamField("width", "Width", Widget.NUMBER, 1024, min=256, max=2048, step=64),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=256, max=2048, step=64),
        # Z-Image-Turbo is distilled: ~8 steps, CFG off (guidance 0). See the model card.
        ParamField("steps", "Steps", Widget.NUMBER, 8, min=1, max=100, step=1),
        ParamField("guidance", "Guidance (CFG)", Widget.NUMBER, 0.0, min=0.0, max=20.0, step=0.5),
        # img2img only: how far to move from the input image (0 = keep, 1 = ignore).
        ParamField(
            "strength", "Denoise strength", Widget.NUMBER, 0.6, min=0.0, max=1.0, step=0.05,
            advanced=True,
        ),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
        # Advanced, optional: pick a specific diffusion file. "" = auto (the single file found under
        # models/diffusion_models/). Lives behind the Adjust panel so the node stays one-click.
        ParamField(
            "model", "Model file (auto)", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        ),
    ),
)


def register_zimage(registry: Any, store: TakeStore, policy: DevicePolicy) -> None:
    """Register the Z-Image node and its runner. Called best-effort by server.bootstrap."""
    registry.register(ZIMAGE, ZImageRunner(store, policy))


class ZImageRunner(NodeRunner):
    produces_takes = True

    def __init__(self, store: TakeStore, policy: DevicePolicy) -> None:
        self._store = store
        self._policy = policy

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        prompt = _first_str(inputs.get("prompt"))
        if not prompt:
            raise ComponentError("Z-Image needs a prompt.")
        params = {**ZIMAGE.defaults(), **node.params}
        width, height = int(params["width"]), int(params["height"])
        steps = max(1, int(params["steps"]))
        guidance = float(params["guidance"])
        negative = str(params.get("negative_prompt") or "").strip() or None
        seed = _resolve_seed(params.get("seed"))
        image_ref = _first(inputs.get("image"))
        img2img = image_ref is not None

        # No hidden downloads: if a required model isn't on disk, fail fast with a message that
        # points at the node's model popup instead of letting diffusers silently fetch it.
        missing = [c.label for c in reqs.zimage_requirements(params) if not c.present]
        if missing:
            raise ComponentError(
                "Z-Image models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )
        resolved = reqs.resolve_diffusion(params)
        if resolved is None:  # defensive: the missing-check above already covers this
            raise ComponentError("Z-Image diffusion model not found in models/diffusion_models/.")
        source, mode = resolved

        ctx.emitter.emit(_progress(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        pipe = _load_pipeline(self._policy, img2img=img2img, source=source, mode=mode)

        placement = self._policy.placement("denoiser")
        gen_device = "cpu" if (placement.offload or self._policy.profile is Profile.CPU) else str(
            placement.device
        )
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        def on_step_end(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if ctx.cancel.cancelled:
                raise CancelledError("Run cancelled.")
            done = step + 1
            ctx.emitter.emit(
                _progress(
                    ctx,
                    node,
                    Phase.SAMPLE,
                    done / steps,
                    step=done,
                    step_count=steps,
                    status=f"Step {done}/{steps}",
                )
            )
            return kwargs

        call: dict[str, Any] = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
        )
        if negative is not None:
            call["negative_prompt"] = negative
        if img2img:
            call["image"] = _load_image(image_ref)
            call["strength"] = float(params.get("strength", 0.6))

        image = pipe(**call).images[0]

        ctx.emitter.emit(_progress(ctx, node, Phase.SAVE, 1.0, status="Saving…"))
        take = self._store.save(
            ctx.run_id,
            node.id,
            image,
            {
                "model": source,
                "prompt": prompt,
                "negative_prompt": negative or "",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                **({"strength": call["strength"]} if img2img else {}),
            },
        )
        return NodeResult(outputs={"image": take}, takes=[take])


# --- pipeline cache -----------------------------------------------------------------------------

# Keyed by (model source, img2img). Built once; diffusers pipelines are not thread-safe, but the run
# manager executes one run at a time (workers=1). The lock guards concurrent first-time builds.
_PIPELINES: dict[tuple[str, bool], Any] = {}
_LOCK = Lock()


def _load_pipeline(policy: DevicePolicy, *, img2img: bool, source: str, mode: str) -> Any:
    key = (source, img2img)
    with _LOCK:
        cached = _PIPELINES.get(key)
        if cached is not None:
            return cached
        # An img2img pipe can reuse the base pipe's already-placed weights (no second load).
        base = _PIPELINES.get((source, False))
        if img2img and base is not None:
            pipe = ZImageImg2ImgPipeline.from_pipe(base)
        else:
            dtype = _torch_dtype(policy.placement("denoiser"))
            pipe = _build_pipeline(source, mode=mode, img2img=img2img, dtype=dtype)
            _configure(pipe, policy)
        _PIPELINES[key] = pipe
        return pipe


def _build_pipeline(source: str, *, mode: str, img2img: bool, dtype: Any) -> Any:
    """Build a Z-Image pipeline **offline** — never touching the network.

    Two shapes, both resolved from files under ``models/`` (see ``requirements.py``):
      - ``mode == "pipeline"``: ``source`` is a whole diffusers folder (``model_index.json`` + all
        components). Loaded with ``local_files_only=True``.
      - ``mode == "single_file"``: ``source`` is a lone diffusion transformer file. We load the
        transformer from it and assemble the pipeline from a **local** VAE + text-encoder +
        tokenizer (present is guaranteed by the missing-check) plus a default flow-match scheduler.

    Drop-in / override: a VAE in ``vae/`` (a ``.safetensors`` or a diffusers dir), an HF-format
    text-encoder dir in ``text_encoders/`` (a bare weights file can't carry its config), or point
    ``INLINE_ZIMAGE_VAE`` / ``INLINE_ZIMAGE_TEXT_ENCODER`` at them. Nothing is fetched — get missing
    pieces via the node's model popup, which writes them under ``models/``.
    """
    cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
    if mode == "pipeline":
        return cls.from_pretrained(source, torch_dtype=dtype, local_files_only=True)

    transformer = ZImageTransformer2DModel.from_single_file(
        source, torch_dtype=dtype, local_files_only=True
    )
    vae = _load_local_vae(dtype)
    text = _load_local_text_encoder(dtype)
    if vae is None or text is None:
        raise ComponentError(
            "Z-Image needs a local VAE and text-encoder for a single-file diffusion model. "
            "Download them from the node's model popup."
        )
    text_encoder, tokenizer = text
    scheduler = _load_scheduler()
    return cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )


def _load_scheduler() -> Any:
    """The flow-match scheduler: from a local pipeline folder's ``scheduler/`` if one exists, else
    the library default. Config-only, so this never downloads."""
    pipe_dir = reqs.pipeline_dir(reqs.diffusion_root())
    if pipe_dir is not None:
        try:
            return FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(pipe_dir), subfolder="scheduler", local_files_only=True
            )
        except (OSError, ValueError):
            pass
    return FlowMatchEulerDiscreteScheduler()


def _load_local_vae(dtype: Any) -> Any:
    path = reqs.local_component("vae", "INLINE_ZIMAGE_VAE")
    if path is None:
        return None
    from diffusers import AutoencoderKL

    if path.is_file():
        return AutoencoderKL.from_single_file(str(path), torch_dtype=dtype, local_files_only=True)
    return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)


def _load_local_text_encoder(dtype: Any) -> tuple[Any, Any] | None:
    path = reqs.local_component("text_encoders", "INLINE_ZIMAGE_TEXT_ENCODER")
    if path is None or not path.is_dir():  # transformers needs a config dir, not a bare file
        return None
    from transformers import AutoModel, AutoTokenizer

    text_encoder = AutoModel.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return text_encoder, tokenizer


def _configure(pipe: Any, policy: DevicePolicy) -> None:
    placement = policy.placement("denoiser")
    if placement.offload:
        pipe.enable_model_cpu_offload()  # opt-in only: stream modules on/off the GPU
    else:
        pipe.to(str(placement.device))  # default: weights resident on the GPU
    # Low-VRAM savers that keep weights on the GPU (no offload): slice attention + tile/slice VAE.
    if policy.attention_slicing():
        _try(pipe.enable_attention_slicing)
    if policy.vae_tiling():
        _try(pipe.enable_vae_tiling)
        _try(pipe.enable_vae_slicing)
    _configure_gpu_speed(pipe, placement)
    _try(pipe.set_progress_bar_config, disable=True)


def _configure_gpu_speed(pipe: Any, placement: Placement) -> None:
    """Throughput tweaks that only apply on a resident-GPU placement (never CPU/offload).

    ``channels_last`` on the conv-based VAE is a safe, default-on win. ``torch.compile`` (the
    transformer) and xformers attention are opt-in via ``INLINE_COMPILE`` / ``INLINE_XFORMERS`` —
    both help but have trade-offs (compile warmup, an extra dep), so they stay off by default. The
    pipeline is cached warm across runs, so a compile cost is paid once. Best-effort."""
    if placement.offload or placement.device.kind is not DeviceKind.CUDA:
        return
    _try(lambda: pipe.vae.to(memory_format=torch.channels_last))
    if os.environ.get("INLINE_XFORMERS") == "1":
        _try(pipe.enable_xformers_memory_efficient_attention)
    if os.environ.get("INLINE_COMPILE") == "1":
        _try(lambda: setattr(pipe, "transformer", torch.compile(pipe.transformer)))


def _torch_dtype(placement: Placement) -> Any:
    return {
        DType.FP16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.FP32: torch.float32,
    }.get(placement.dtype, torch.bfloat16)


# --- small helpers ------------------------------------------------------------------------------


def _resolve_seed(raw: Any) -> int:
    """A fixed non-negative seed passes through; -1 (or anything invalid) becomes a fresh random."""
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        seed = -1
    return seed if seed >= 0 else random.randint(0, _SEED_MAX)


def _load_image(ref: Any) -> Any:
    from PIL import Image

    if isinstance(ref, AssetRef) and ref.ref == "path" and ref.path:
        return Image.open(ref.path).convert("RGB")
    raise ComponentError("Z-Image img2img needs a readable image path input.")


def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


def _first_str(values: list[Any] | None) -> str:
    value = _first(values)
    return str(value).strip() if value is not None else ""


def _progress(
    ctx: ExecutionContext,
    node: Node,
    phase: Phase,
    fraction: float,
    *,
    step: int | None = None,
    step_count: int | None = None,
    status: str = "",
) -> ProgressEvent:
    return ProgressEvent(
        run_id=ctx.run_id,
        node_id=node.id,
        phase=phase,
        fraction=fraction,
        step=step,
        step_count=step_count,
        status=status,
    )


def _try(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Best-effort optional pipeline tweak; skip if this build lacks it or the op isn't supported
    here (e.g. xformers not installed, torch.compile unavailable on this device)."""
    try:
        fn(*args, **kwargs)
    except (AttributeError, ValueError, NotImplementedError, RuntimeError, TypeError, ImportError):
        pass
