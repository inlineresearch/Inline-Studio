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
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline, ZImageTransformer2DModel

from ...config import models_dir
from ...device.policy import DevicePolicy, Placement, Profile
from ...device.types import DType
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase, ProgressEvent
from ...runtime.store import TakeStore
from ...takes import AssetRef

# One model file, everything else behind the scenes. Users drop a single Z-Image diffusion
# `.safetensors` into models/diffusion_models/ (the ComfyUI-style file). We load the transformer
# from that file and assemble the VAE / text-encoder / tokenizer / scheduler from the reference repo
# (a one-time cached fetch of those small components — not the multi-GB diffusion weights). No repo
# folder to set up, no low-level load nodes. Override the source with INLINE_ZIMAGE_MODEL (a file
# path, a local diffusers dir, or an HF repo id) or the node's advanced `model` param.
_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"  # supplies vae/text-encoder/tokenizer/scheduler + configs
_DEFAULT_MODEL = _BASE_REPO
_LOCAL_NAMES = ("Z-Image-Turbo", "z-image-turbo", "Z-Image", "z-image")
_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".sft")

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

        source, single_file = _resolve_model(params)
        ctx.emitter.emit(
            _progress(ctx, node, Phase.LOADING, 0.0, status="Loading Z-Image")
        )
        pipe = _load_pipeline(self._policy, img2img=img2img, source=source, single_file=single_file)

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
                _progress(ctx, node, Phase.SAMPLE, done / steps, step=done, step_count=steps)
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

        ctx.emitter.emit(_progress(ctx, node, Phase.SAVE, 1.0))
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


def _load_pipeline(policy: DevicePolicy, *, img2img: bool, source: str, single_file: bool) -> Any:
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
            pipe = _build_pipeline(source, single_file=single_file, img2img=img2img, dtype=dtype)
            _configure(pipe, policy)
        _PIPELINES[key] = pipe
        return pipe


def _build_pipeline(source: str, *, single_file: bool, img2img: bool, dtype: Any) -> Any:
    """Build a Z-Image pipeline. A diffusers dir / repo id loads whole; a single `.safetensors` is
    the diffusion transformer only: we load it from the file and take the remaining components
    (vae, text-encoder, tokenizer, scheduler) from local files when present, else the ref repo.

    Bring-your-own / offline: drop the diffusion `.safetensors` in ``diffusion_models/``, a VAE
    in ``vae/`` (a single ``.safetensors`` or a diffusers dir), and an HF-format text-encoder dir in
    ``text_encoders/`` (a bare weights file can't carry its config, so a dir is required there). Any
    component you don't provide is fetched from the reference repo (network once). Override the
    paths with ``INLINE_ZIMAGE_VAE`` / ``INLINE_ZIMAGE_TEXT_ENCODER``.
    """
    cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
    if not single_file:
        return cls.from_pretrained(source, torch_dtype=dtype)
    components: dict[str, Any] = {
        "transformer": ZImageTransformer2DModel.from_single_file(source, torch_dtype=dtype)
    }
    vae = _load_local_vae(dtype)
    if vae is not None:
        components["vae"] = vae
    text = _load_local_text_encoder(dtype)
    if text is not None:
        components["text_encoder"], components["tokenizer"] = text
    return cls.from_pretrained(_BASE_REPO, torch_dtype=dtype, **components)


def _local_component(category: str, env_var: str) -> Path | None:
    """A local supporting-model file/dir under ``models/<category>/`` (or env override), or None.
    Prefers an explicit env path, then a single weight file, then a subdir (HF snapshot)."""
    env = os.environ.get(env_var, "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    root = models_dir() / category
    if not root.is_dir():
        return None
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )
    if files:
        return files[0]
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    return dirs[0] if dirs else None


def _load_local_vae(dtype: Any) -> Any:
    path = _local_component("vae", "INLINE_ZIMAGE_VAE")
    if path is None:
        return None
    from diffusers import AutoencoderKL

    if path.is_file():
        return AutoencoderKL.from_single_file(str(path), torch_dtype=dtype)
    return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype)


def _load_local_text_encoder(dtype: Any) -> tuple[Any, Any] | None:
    path = _local_component("text_encoders", "INLINE_ZIMAGE_TEXT_ENCODER")
    if path is None or not path.is_dir():  # transformers needs a config dir, not a bare file
        return None
    from transformers import AutoModel, AutoTokenizer

    text_encoder = AutoModel.from_pretrained(str(path), torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
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
    _try(pipe.set_progress_bar_config, disable=True)


def _resolve_model(params: dict[str, Any] | None = None) -> tuple[str, bool]:
    """Pick the Z-Image source. Returns ``(source, single_file)`` where ``single_file`` means a lone
    diffusion `.safetensors` (transformer only). Priority, most specific first:
      1. the node's advanced ``model`` param — a filename under models/diffusion_models/;
      2. ``INLINE_ZIMAGE_MODEL`` — a file path, a local diffusers dir, or an HF repo id;
      3. a single weight file dropped under models/diffusion_models/ (the common, zero-config case);
      4. a local diffusers folder (model_index.json);
      5. the default reference repo.
    """
    root = models_dir() / "diffusion_models"

    chosen = str((params or {}).get("model") or "").strip()
    if chosen:
        path = root / chosen
        if path.is_file():
            return str(path), True

    env = os.environ.get("INLINE_ZIMAGE_MODEL", "").strip()
    if env:
        return (env, True) if Path(env).is_file() else (env, False)

    single = _find_weight_file(root)
    if single is not None:
        return str(single), True

    for name in _LOCAL_NAMES:
        candidate = root / name
        if (candidate / "model_index.json").is_file():
            return str(candidate), False

    return _DEFAULT_MODEL, False


def _find_weight_file(root: Path) -> Path | None:
    """The single diffusion weight file to load: prefer a z-image-named file, else the first one."""
    if not root.is_dir():
        return None
    weights = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )
    named = [p for p in weights if "z" in p.name.lower() and "image" in p.name.lower()]
    return (named or weights or [None])[0]


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
    """Best-effort optional pipeline tweak; skip if this diffusers build lacks it."""
    try:
        fn(*args, **kwargs)
    except (AttributeError, ValueError, NotImplementedError):
        pass
