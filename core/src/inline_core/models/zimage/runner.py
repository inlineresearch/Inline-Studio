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

import logging
import os
import random
import time
from threading import Lock
from typing import Any

import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

from ...device.policy import DevicePolicy, OffloadMode, Placement, Profile, Quantization
from ...device.types import DeviceKind, DType
from ...errors import CancelledError, ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.loader_runners import ComponentRef
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...media import MediaKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase, ProgressEvent
from ...runtime.store import TakeStore
from ...takes import AssetRef
from .. import loaders
from . import requirements as reqs

# The models this node needs — the diffusion transformer plus a VAE, text-encoder, tokenizer and
# scheduler — are assembled entirely from files under models/ (see `requirements.py`). **Nothing is
# ever downloaded here.** Every diffusers/transformers load below runs with local_files_only=True,
# so a missing model is a clear error pointing at the node's model popup — never a silent fetch from
# Hugging Face. Models arrive by exactly two paths: the user drops files under models/, or the popup
# downloads them into models/.
_SEED_MAX = 2**31 - 1

logger = logging.getLogger("inline_core.zimage")


def _device_report(policy: DevicePolicy) -> str:
    """A one-line snapshot of where generation will run — device, dtype, cpu/gpu mode, and (on
    CUDA) live VRAM used/total/free. Logged around model load + generation so a slow run's cause
    (CPU fallback, offload, a nearly-full GPU) is visible without a profiler."""
    placement = policy.placement("denoiser")
    device = str(placement.device)
    on_cpu = placement.offload or policy.profile is Profile.CPU
    parts = [
        f"device={device}",
        f"mode={'cpu' if on_cpu else 'gpu'}",
        f"profile={policy.profile.value}",
        f"dtype={placement.dtype.value}",
        f"offload={placement.offload_mode.value}",
        f"quant={policy.quantization().value}",
    ]
    if placement.device.kind is DeviceKind.CUDA and torch.cuda.is_available():
        idx = placement.device.index or 0
        try:
            free, total = torch.cuda.mem_get_info(idx)
            used = torch.cuda.memory_allocated(idx)
            parts.append(f"gpu={torch.cuda.get_device_name(idx)}")
            parts.append(
                f"vram={used / 1e9:.1f}GB allocated / {total / 1e9:.1f}GB total"
                f" ({free / 1e9:.1f}GB free)"
            )
        except Exception:  # noqa: BLE001 — diagnostics must never break a generation
            pass
    return ", ".join(parts)


ZIMAGE = NodeDescriptor(
    type="alibaba/z-image-turbo",
    title="Z-Image Turbo",
    category="Generate",
    icon="wand",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
        # Optional component handles from load/* subnodes — wire a Load node to override the
        # corresponding dropdown. Left unwired, the node resolves each file from its own selects.
        Port("model", "Diffusion model", PortKind.MODEL, required=False),
        Port("vae", "VAE", PortKind.VAE, required=False),
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=False),
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
        # Advanced, optional: pick a specific file per component. "" = auto (the single file in that
        # category folder). ComfyUI-style split-file loading — one file each. These live behind the
        # Adjust panel so the node stays one-click; the model popup downloads the defaults.
        ParamField(
            "model", "Diffusion file (auto)", Widget.SELECT, "",
            options_from="diffusion_models", advanced=True,
        ),
        ParamField(
            "text_encoder", "Text-encoder file (auto)", Widget.SELECT, "",
            options_from="text_encoders", advanced=True,
        ),
        ParamField(
            "vae", "VAE file (auto)", Widget.SELECT, "",
            options_from="vae", advanced=True,
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

        # Optional wired component handles from load/* subnodes override the dropdowns.
        model_ref = _component_ref(inputs, "model", "diffusion")
        vae_ref = _component_ref(inputs, "vae", "vae")
        te_ref = _component_ref(inputs, "text_encoder", "text_encoder")
        wired = {ref.kind for ref in (model_ref, vae_ref, te_ref) if ref is not None}

        # No hidden downloads: a required component that is neither wired nor on disk fails fast,
        # pointing at the model popup — never a silent diffusers fetch. Wired components are present
        # by construction (the Load node resolved a real file, or raised).
        missing = [
            c.label for c in reqs.zimage_requirements(params) if not c.present and c.id not in wired
        ]
        if missing:
            raise ComponentError(
                "Z-Image models missing: "
                + ", ".join(missing)
                + ". Download them from the node's model popup (the hint on the node)."
            )

        if model_ref is not None:
            mode, source = "single_file", model_ref.file
        else:
            resolved = reqs.resolve_diffusion(params)
            if resolved is None:  # defensive: the missing-check above already covers this
                raise ComponentError("Z-Image diffusion model not found in diffusion_models/.")
            mode, source = resolved  # resolve_diffusion returns (mode, path)
        # In single-file mode the VAE + text-encoder are their own chosen files (a wired handle, or
        # the dropdown / split file); in whole-pipeline mode the folder carries them, so these go
        # unused unless explicitly wired.
        vae_file = vae_ref.file if vae_ref else _path_or_none(reqs.resolve_vae(params))
        te_file = te_ref.file if te_ref else _path_or_none(reqs.resolve_text_encoder(params))

        logger.info(
            "Z-Image run: %dx%d, %d steps, guidance=%.1f, img2img=%s | %s",
            width,
            height,
            steps,
            guidance,
            img2img,
            _device_report(self._policy),
        )
        _reset_peak_vram()
        ctx.emitter.emit(_progress(ctx, node, Phase.LOADING, 0.0, status="Loading model…"))
        try:
            pipe = _load_pipeline(
                self._policy,
                img2img=img2img,
                source=source,
                mode=mode,
                vae=vae_file,
                text=te_file,
                quant=self._policy.quantization(),
            )
        except torch.cuda.OutOfMemoryError as error:
            _free_vram()
            raise ComponentError(_oom_message(width, height)) from error

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

        logger.info("Z-Image sampling %d steps on %s…", steps, gen_device)
        sample_start = time.perf_counter()
        try:
            image = pipe(**call).images[0]
        except torch.cuda.OutOfMemoryError as error:
            _free_vram()
            raise ComponentError(_oom_message(width, height)) from error
        elapsed = time.perf_counter() - sample_start
        peak_gb = _peak_vram_gb()
        peak_note = f", peak VRAM {peak_gb:.1f}GB" if peak_gb else ""
        logger.info(
            "Z-Image sampled %dx%d in %.1fs (%.2fs/step)%s | %s",
            width,
            height,
            elapsed,
            elapsed / steps,
            peak_note,
            _device_report(self._policy),
        )
        _free_vram()  # return fragmented free blocks to the driver between runs (keeps the model)

        save_status = "Saving…" + (f" (peak VRAM {peak_gb:.1f}GB)" if peak_gb else "")
        ctx.emitter.emit(_progress(ctx, node, Phase.SAVE, 1.0, status=save_status))
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

# Keyed by (source, vae, text, img2img, quant). Built once; diffusers pipelines are not thread-safe,
# but the run manager executes one run at a time (workers=1). Switching any file or the quantization
# rebuilds — but the loader core caches each component, so only the changed one re-reads from disk.
# The lock guards concurrent first-time builds.
_PIPELINES: dict[tuple[str, str, str, bool, str], Any] = {}
_LOCK = Lock()


def _load_pipeline(
    policy: DevicePolicy,
    *,
    img2img: bool,
    source: str,
    mode: str,
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
) -> Any:
    key = (source, vae, text, img2img, quant.value)
    with _LOCK:
        cached = _PIPELINES.get(key)
        if cached is not None:
            logger.info(
                "Pipeline cache hit (%s, img2img=%s) — reusing loaded weights", source, img2img
            )
            return cached
        started = time.perf_counter()
        logger.info(
            "Loading Z-Image pipeline: source=%s, mode=%s, img2img=%s | %s",
            source,
            mode,
            img2img,
            _device_report(policy),
        )
        # An img2img pipe can reuse the base pipe's already-placed weights (no second load).
        base = _PIPELINES.get((source, vae, text, False, quant.value))
        if img2img and base is not None:
            pipe = ZImageImg2ImgPipeline.from_pipe(base)
            logger.info(
                "Built img2img pipeline from cached base in %.1fs", time.perf_counter() - started
            )
        else:
            dtype = _torch_dtype(policy.placement("denoiser"))
            vae_dtype = _torch_dtype(policy.placement("vae"))
            build_start = time.perf_counter()
            pipe = _build_pipeline(
                source,
                mode=mode,
                img2img=img2img,
                dtype=dtype,
                vae=vae,
                text=text,
                quant=quant,
                vae_dtype=vae_dtype,
            )
            logger.info(
                "Read weights from disk in %.1fs (mode=%s, dtype=%s)",
                time.perf_counter() - build_start,
                mode,
                policy.placement("denoiser").dtype.value,
            )
            place_start = time.perf_counter()
            _configure(pipe, policy)
            logger.info(
                "Placed pipeline on %s in %.1fs | %s",
                str(policy.placement("denoiser").device),
                time.perf_counter() - place_start,
                _device_report(policy),
            )
        _PIPELINES[key] = pipe
        logger.info("Z-Image pipeline ready in %.1fs total", time.perf_counter() - started)
        return pipe


def _build_pipeline(
    source: str,
    *,
    mode: str,
    img2img: bool,
    dtype: Any,
    vae: str,
    text: str,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
) -> Any:
    """Build a Z-Image pipeline **offline** — never touching the network.

    Two shapes, both resolved from files under ``models/`` (see ``requirements.py``):
      - ``mode == "pipeline"``: ``source`` is a whole diffusers folder (``model_index.json`` + all
        components). Loaded with ``local_files_only=True``.
      - ``mode == "single_file"``: ComfyUI-style. ``source`` / ``vae`` / ``text`` are three single
        ``.safetensors`` files; the loader core (``models/loaders.py``) builds each component from a
        bundled config + tokenizer, so nothing is fetched. This is the fast path the docs describe.

    ``quant`` (smart memory) quantizes the big weights on load; it applies to the single-file path
    (the loader builds each component and can quantize it). Whole-pipeline folders load full
    precision — quantize by using the single-file layout instead.
    """
    if mode == "pipeline":
        if quant is not Quantization.NONE:
            logger.warning(
                "Smart-memory quantization (%s) is not applied to a whole-pipeline folder; use the "
                "single-file layout (diffusion_models/ + vae/ + text_encoders/) to quantize.",
                quant.value,
            )
        cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
        return cls.from_pretrained(source, torch_dtype=dtype, local_files_only=True)

    if not vae or not text:
        raise ComponentError(
            "Z-Image needs a local VAE and text-encoder file for a single-file diffusion model. "
            "Download them from the node's model popup."
        )
    return loaders.assemble_zimage_pipeline(
        diffusion_file=source,
        vae_file=vae,
        text_encoder_file=text,
        dtype=dtype,
        img2img=img2img,
        quant=quant,
        vae_dtype=vae_dtype,
    )


def _configure(pipe: Any, policy: DevicePolicy) -> None:
    placement = policy.placement("denoiser")
    if placement.offload_mode is OffloadMode.SEQUENTIAL:
        # Lowest peak VRAM: submodules stream on/off the GPU layer-by-layer. Slowest — only when the
        # GPU is too small even for model offload.
        pipe.enable_sequential_cpu_offload()
    elif placement.offload_mode is OffloadMode.MODEL:
        # Smart memory / opt-in: only the active component sits on the GPU, the rest waits in RAM.
        pipe.enable_model_cpu_offload()
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


# --- VRAM helpers (diagnostics + between-run hygiene; never break a generation) ------------------


def _reset_peak_vram() -> None:
    """Zero CUDA's peak-allocation counter so the value read after a run reflects *this* run."""
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — telemetry must never break a generation
        pass


def _peak_vram_gb() -> float:
    """Peak VRAM allocated since the last reset, in GB (0.0 when unavailable/CPU)."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _free_vram() -> None:
    """Release cached-but-unused CUDA blocks back to the driver. Cuts the fragmentation that makes a
    later allocation fail even with headroom; it does NOT evict the resident (cached) model."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _oom_message(width: int, height: int) -> str:
    """A clear, actionable message for a CUDA out-of-memory — never a raw allocator traceback.
    Points at smart memory (unless already on) and the resolution, which drives peak memory."""
    smart = os.environ.get("INLINE_SMART_MEMORY", "").strip().lower() in ("1", "true", "yes", "on")
    if smart:
        tip = f"lower the resolution (you're at {width}x{height} — try 768x768 or 512x512)"
    else:
        tip = "enable smart memory (run ./webui.sh --smart-memory) or lower the resolution"
    return f"Ran out of GPU memory generating a {width}x{height} image. Try to {tip}."


# --- small helpers ------------------------------------------------------------------------------


def _component_ref(inputs: dict[str, list[Any]], port: str, kind: str) -> ComponentRef | None:
    """A wired ``ComponentRef`` on ``port``, or None if unwired. Guards the kind so a mis-wired
    handle (e.g. a VAE fed into the model port — which the graph validator already blocks by port
    kind) can't slip a wrong component into the pipeline."""
    ref = _first(inputs.get(port))
    if ref is None:
        return None
    if isinstance(ref, ComponentRef) and ref.kind == kind:
        return ref
    raise ComponentError(f"Z-Image '{port}' input is not a loadable {kind} handle.")


def _path_or_none(path: Any) -> str:
    """A resolved component path as a string, or ``""`` when absent — the pipeline cache key and the
    single-file check both want a plain string, not ``Path | None``."""
    return str(path) if path is not None else ""


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
