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
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any, cast

import torch
from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

from ...device.policy import (
    DevicePolicy,
    FitEstimate,
    ModelFootprint,
    OffloadMode,
    Placement,
    Profile,
    Quantization,
)
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
from ..sampling import SamplingFamily, apply_sampling, sampling_param_fields
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
        # Sampler + scheduler (advanced): the two SELECT dropdowns from the reusable sampling
        # registry. Z-Image is flow-match, so these tune the FlowMatchEuler scheduler (ancestral +
        # sigma spacing) rather than swapping sampler classes — see models/sampling.py.
        *sampling_param_fields(SamplingFamily.FLOW_MATCH),
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
        sampler = str(params["sampler"])
        scheduler = str(params["scheduler"])
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

        # Size-aware placement: hand the policy the model's on-disk sizes so it fits dtype/quant/
        # offload to THIS GPU — int8 auto-engages on a card too small for full precision (a T4),
        # with no --smart-memory flag. Then refuse an impossible load up front, so a too-big model
        # is a clean node error instead of a host-RAM OOM-kill that takes the whole server down.
        self._policy.set_footprint(_footprint(mode, source, vae_file, te_file))
        fit = self._policy.fit_estimate()
        if fit is not None and not fit.fits:
            raise ComponentError(_wont_fit_message(fit))

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
        except MemoryError as error:
            # Host-RAM exhaustion (a Python MemoryError) — surfaced cleanly. The streaming loaders
            # keep peak RAM ≈ one tensor, so this is the rare tail; an OS OOM-kill (SIGKILL) cannot
            # be caught here — the pre-flight check above is what prevents ever reaching that spike.
            _free_vram()
            raise ComponentError(_oom_message(width, height, host=True)) from error

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
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
        )
        # Encode the prompt and (on a resident GPU) drop the text encoder to CPU so its VRAM is free
        # for the denoise — otherwise it sits idle on the card and OOMs a tight GPU (a 16 GB T4).
        call.update(
            _prompt_kwargs(pipe, self._policy, prompt=prompt, negative=negative, guidance=guidance)
        )
        if img2img:
            call["image"] = _load_image(image_ref)
            call["strength"] = float(params.get("strength", 0.6))

        # Rebuild the scheduler for the chosen sampler/scheduler from the pipe's pristine base
        # config (immutable across cache hits). `uniform` returns an explicit sigmas array; the rest
        # flip a FlowMatchEuler config flag and return None.
        base_config = getattr(pipe, "_inline_base_scheduler_config", None)
        if base_config is not None:
            sigmas = apply_sampling(
                pipe, base_config, SamplingFamily.FLOW_MATCH, sampler, scheduler, steps
            )
            if sigmas is not None:
                call["sigmas"] = sigmas

        logger.info(
            "Z-Image sampling %d steps on %s (sampler=%s, scheduler=%s)…",
            steps,
            gen_device,
            sampler,
            scheduler,
        )
        sample_start = time.perf_counter()
        try:
            with _text_encoder_detached(pipe, "prompt_embeds" in call):
                image = pipe(**call).images[0]
        except torch.cuda.OutOfMemoryError as error:
            _free_vram()
            raise ComponentError(_oom_message(width, height)) from error
        except MemoryError as error:
            _free_vram()
            raise ComponentError(_oom_message(width, height, host=True)) from error
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
                "sampler": sampler,
                "scheduler": scheduler,
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
        # Free any *other* model still resident before loading this one, so switching checkpoints
        # doesn't stack VRAM/RAM (the caches never evicted before). Keeps the current source's
        # components — including a base pipeline reused below for img2img.
        _evict_stale(source, vae, text)
        # An img2img pipe can reuse the base pipe's already-placed weights (no second load).
        base = _PIPELINES.get((source, vae, text, False, quant.value))
        if img2img and base is not None:
            pipe = ZImageImg2ImgPipeline.from_pipe(base)
            logger.info(
                "Built img2img pipeline from cached base in %.1fs", time.perf_counter() - started
            )
        else:
            placement = policy.placement("denoiser")
            dtype = _torch_dtype(placement)
            vae_dtype = _torch_dtype(policy.placement("vae"))
            # Resident placement streams weights straight to the GPU during load (no CPU copy); the
            # offload path loads to CPU so accelerate can install its hooks before placing.
            load_device = None if placement.offload else str(placement.device)
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
                device=load_device,
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
        _capture_base_scheduler_config(pipe, base if img2img else None)
        _PIPELINES[key] = pipe
        logger.info("Z-Image pipeline ready in %.1fs total", time.perf_counter() - started)
        return pipe


def _capture_base_scheduler_config(pipe: Any, base: Any) -> None:
    """Stash a pipe's ORIGINAL scheduler config once, so every run rebuilds the scheduler from an
    immutable snapshot instead of a prior run's mutated one.

    The ``_PIPELINES`` cache key does not include sampler/scheduler, and ``apply_sampling`` replaces
    ``pipe.scheduler`` per run — so without a pristine snapshot a later run would rebuild on top of
    an earlier selection's config (a stale-scheduler leak across cache hits). Captured here at build
    time, before any run mutates the pipe. An img2img pipe built via ``from_pipe`` inherits the base
    pipe's pristine snapshot (the base's own ``scheduler`` may already be swapped by a run)."""
    if getattr(pipe, "_inline_base_scheduler_config", None) is not None:
        return
    inherited = getattr(base, "_inline_base_scheduler_config", None) if base is not None else None
    if inherited is not None:
        pipe._inline_base_scheduler_config = inherited
        return
    scheduler = getattr(pipe, "scheduler", None)
    pipe._inline_base_scheduler_config = dict(scheduler.config) if scheduler is not None else None


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
    device: str | None = None,
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
        device=device,
    )


def _evict_stale(source: str, vae: str, text: str) -> None:
    """Free every *other* model's pipelines + components before loading a new checkpoint, so a
    second distinct model doesn't stack on the first. Entries for the current ``(source, vae,
    text)`` — e.g. a cached base pipeline reused for img2img — are kept. Called under ``_LOCK``."""
    import gc

    keep_triple = (source, vae, text)
    for k in list(_PIPELINES):
        if k[:3] == keep_triple:
            continue
        del _PIPELINES[k]
    gc.collect()
    loaders.unload_components(keep_files={source, vae, text})
    _free_vram()


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
    # Wrap each in a lambda so a pipeline that lacks the method (e.g. ZImagePipeline has no
    # ``enable_vae_tiling``) is skipped by ``_try`` — a bare ``_try(pipe.enable_vae_tiling)`` would
    # raise AttributeError while evaluating the argument, before ``_try`` could swallow it.
    if policy.attention_slicing():
        _try(lambda: pipe.enable_attention_slicing())
    if policy.vae_tiling():
        # ZImagePipeline exposes NO ``enable_vae_tiling``/``enable_vae_slicing`` — those calls are
        # silent no-ops, so a 1024² VAE decode would run un-tiled (full-frame conv activations, a
        # multi-GB VRAM spike). The underlying AutoencoderKL DOES expose the real methods; call them
        # on the VAE directly so tiling actually engages.
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            _try(lambda: vae.enable_tiling())
            _try(lambda: vae.enable_slicing())
            _shrink_vae_tiles(vae)
    _configure_gpu_speed(pipe, placement)
    _try(pipe.set_progress_bar_config, disable=True)


def _shrink_vae_tiles(vae: Any, tile_px: int = 512) -> None:
    """Force the VAE decode to actually tile at 1024².

    ``AutoencoderKL._decode`` only tiles when the latent is **strictly larger** than
    ``tile_latent_min_size`` — which defaults to ``sample_size / spatial_scale`` (128 for this VAE,
    an 8× downscale of a 1024 sample). A 1024² image has a latent of exactly 128, so ``128 > 128``
    is False: tiling silently does NOT engage even after ``enable_tiling()``, and full-frame decode
    allocates a single ~4.5 GB conv activation that OOMs a T4. Shrinking the tile to ``tile_px``
    (512, → a latent threshold of 64) makes 1024 (and up) decode in 512-px tiles, cutting the peak
    ~4× for a light seam-blend cost. Best-effort: leaves the VAE untouched if it lacks the attrs."""
    sample = getattr(vae, "tile_sample_min_size", None)
    latent = getattr(vae, "tile_latent_min_size", None)
    if not sample or not latent or tile_px >= int(sample):
        return
    scale = int(sample) / int(latent)  # the VAE's spatial downscale (8)
    vae.tile_sample_min_size = tile_px
    vae.tile_latent_min_size = max(1, int(tile_px / scale))


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


# --- text-encoder offload (reclaim its VRAM for the denoise) -------------------------------------


def _supports_prompt_embeds(pipe: Any) -> bool:
    """True when this pipeline can take precomputed prompt embeddings, so we can encode the prompt,
    free the text encoder from the GPU, then denoise. Guards the offload against a diffusers version
    whose ``__call__`` lacks the parameter (then we keep the raw-prompt path)."""
    if not hasattr(pipe, "encode_prompt"):
        return False
    try:
        import inspect

        return "prompt_embeds" in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False


@contextmanager
def _text_encoder_detached(pipe: Any, active: bool) -> Iterator[None]:
    """Temporarily remove the text encoder from the pipeline for the denoise, then restore it.

    Why: when we pre-encode and park the encoder on the CPU, diffusers' ``_execution_device`` (→
    ``DiffusionPipeline.device``) picks the device of *some* registered nn.Module — and it iterates
    a **set** of module names, so the pick is non-deterministic. If it lands on the CPU-parked
    encoder, ``prepare_latents`` builds the latents on the CPU while our generator + transformer are
    on CUDA → "Cannot generate a cpu tensor from a generator of type cuda", or a device mismatch.

    Since we hand the pipeline precomputed ``prompt_embeds``, ``__call__`` never touches the text
    encoder — so detaching it for the call leaves only CUDA modules (vae + transformer) for device
    inference, making the execution device deterministically the GPU. Restored in ``finally`` so the
    cached pipeline can encode again next run. ``active`` is False on the raw-prompt path (encoder
    still needed), where this is a no-op."""
    if not active:
        yield
        return
    saved = getattr(pipe, "text_encoder", None)
    pipe.text_encoder = None
    try:
        yield
    finally:
        pipe.text_encoder = saved


def _raw_prompt_kwargs(prompt: str, negative: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"prompt": prompt}
    if negative is not None:
        kwargs["negative_prompt"] = negative
    return kwargs


def _prompt_kwargs(
    pipe: Any, policy: DevicePolicy, *, prompt: str, negative: str | None, guidance: float
) -> dict[str, Any]:
    """The prompt argument(s) for the pipeline call.

    On a **resident** GPU placement the text encoder (Qwen3-4B — ~4 GB even int8) otherwise sits
    idle on the card through the entire denoise, starving the sampler + VAE decode of VRAM. So we
    pre-encode the prompt, then park the encoder on the CPU, and hand the pipeline precomputed
    embeddings — the transformer + VAE stay resident (fast), the denoise gets the encoder's ~4 GB.

    Crucially we encode **on the GPU**, not the CPU. torchao weight-only int8 has a real CUDA matmul
    kernel: it dequantizes each weight to bf16 transiently, per-op, and CUDA frees it immediately —
    the forward barely moves peak VRAM (measured: <1 GB over the ~11 GB resident, well inside the
    ~4 GB free on a T4). The CPU has **no** int8 matmul kernel, so a CPU encode instead dequantizes
    the *entire* Qwen3-4B to bf16 in host RAM (~8 GB) and, on a 16 GB box with no swap, the OS
    OOM-kills the whole server mid-encode — the exact crash in scripts/mem1024.log (host RAM
    climbed 2.9 → 16.1 GB while VRAM sat flat). So: encode with the encoder resident, THEN move it
    to the CPU to reclaim its VRAM. Parking is a plain tensor copy — no forward, no dequant — ~4 GB
    of (plentiful) host RAM, no spike. On an offload/CPU placement accelerate already streams the
    encoder, so we pass the raw prompt.

    Best-effort: any failure falls back to the raw-prompt path (today's behavior) so a diffusers
    change can never turn a working generation into a hard error."""
    placement = policy.placement("denoiser")
    resident = (
        placement.device.kind is DeviceKind.CUDA
        and not placement.offload
        and policy.profile is not Profile.CPU
    )
    text_encoder = getattr(pipe, "text_encoder", None)
    if not resident or text_encoder is None or not _supports_prompt_embeds(pipe):
        return _raw_prompt_kwargs(prompt, negative)

    device = str(placement.device)
    do_cfg = guidance > 0
    try:
        logger.info(
            "Encoding prompt on %s (no_grad) | host RAM %.1fGB | %s",
            device,
            _host_ram_gb(),
            _device_report(policy),
        )
        text_encoder.to(device)  # ensure the encoder is on the card for a GPU (int8-kernel) encode
        # ``encode_prompt`` called directly is NOT wrapped in the pipeline's ``@torch.no_grad``
        # (only ``__call__`` is), so without this the Qwen forward runs with autograd ON and keeps
        # the full activation graph across all 36 layers — ~8 GB on the T4. That graph (not the int8
        # dequant) is what spiked host RAM to 16 GB on a CPU encode (OOM-kill) and needed the extra
        # ~1.9 GB that OOMed a GPU encode. no_grad (what ``__call__`` uses) drops it: the forward
        # frees each layer's activations as it goes, so the encode fits in the ~4 GB free.
        with torch.no_grad():
            prompt_embeds, negative_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative,
                do_classifier_free_guidance=do_cfg,
                device=torch.device(device),
            )
        # Reclaim the encoder's ~4 GB of VRAM for the denoise. A plain copy to CPU — no forward
        # there, so no int8->bf16 host-RAM dequant spike (the crash we first hit). We do NOT touch
        # the int8 transformer: torchao's ``.to()`` round-trip on quantized weights is unreliable
        # (it strands the weight on the wrong device); keep it resident, get headroom from no_grad.
        text_encoder.to("cpu")
        _free_vram()
        # Embeds already sit on ``device`` (we passed it to encode_prompt); this normalizes list vs
        # tensor and is a no-op move when already placed.
        prompt_embeds = _embeds_to(prompt_embeds, device)
        if do_cfg:
            negative_embeds = _embeds_to(negative_embeds, device)
    except Exception as error:  # noqa: BLE001 — an optimization must never break generation
        logger.warning(
            "Text-encoder GPU encode failed (%s); denoising with the encoder resident.", error
        )
        _try(text_encoder.to, device)
        return _raw_prompt_kwargs(prompt, negative)

    logger.info(
        "Encoded prompt on the GPU; text encoder parked on the CPU for the denoise | "
        "host RAM %.1fGB | %s",
        _host_ram_gb(),
        _device_report(policy),
    )
    kwargs: dict[str, Any] = {"prompt_embeds": prompt_embeds}
    if do_cfg:
        # __call__ requires the negatives alongside the positives when CFG is on.
        kwargs["negative_prompt_embeds"] = negative_embeds
    return kwargs


def _embeds_to(embeds: Any, device: str) -> Any:
    """Move encoder output onto ``device``. ``ZImagePipeline.encode_prompt`` returns a *list* of
    per-prompt embedding tensors (variable length, one per prompt); a future/other pipeline might
    return a single tensor — handle both."""
    if isinstance(embeds, (list, tuple)):
        return [e.to(device) for e in cast("list[Any]", embeds)]
    return embeds.to(device)


# --- VRAM helpers (diagnostics + between-run hygiene; never break a generation) ------------------


def _host_ram_gb() -> float:
    """Host RAM currently in use (total − available), in GB. A cheap /proc read so a run's log shows
    RAM staying flat through encode/denoise — the signal that the CPU-encode OOM-kill is gone. 0.0
    when /proc/meminfo is unavailable (non-Linux). Never breaks a generation."""
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, float] = {}
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 1e6  # kB -> GB
        avail = info.get("MemAvailable", info.get("MemFree", 0.0))
        return max(0.0, info.get("MemTotal", 0.0) - avail)
    except Exception:  # noqa: BLE001 — telemetry must never break a generation
        return 0.0


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


def _smaller_resolutions(width: int, height: int) -> list[str]:
    """Standard square sizes strictly smaller than the current image, largest first — so the OOM
    hint only ever suggests a resolution that actually cuts peak memory. Falls back to halving the
    current size when nothing on the ladder is smaller (e.g. already at or below 512)."""
    ladder = [1024, 768, 512, 384, 256]
    current = max(width, height)
    smaller = [f"{s}x{s}" for s in ladder if s < current]
    if smaller:
        return smaller[:2]
    half = max(64, (current // 2 // 8) * 8)
    return [f"{half}x{half}"]


def _oom_message(width: int, height: int, *, host: bool = False) -> str:
    """A clear, actionable message for an out-of-memory — never a raw allocator traceback. ``host``
    distinguishes system-RAM exhaustion from GPU VRAM. Points at the resolution (which drives peak
    memory) and a smaller model."""
    if host:
        return (
            f"Ran out of system RAM loading the model for a {width}x{height} image. Close other "
            "apps, use a machine with more RAM, or pick a smaller model."
        )
    suggestions = " or ".join(_smaller_resolutions(width, height))
    return (
        f"Ran out of GPU memory generating a {width}x{height} image. Lower the resolution "
        f"(you're at {width}x{height} — try {suggestions}) or pick a smaller model."
    )


def _footprint(mode: str, source: str, vae: str, text: str) -> ModelFootprint:
    """The model's on-disk component sizes for the fit estimate. Single-file mode only — a whole
    diffusers pipeline folder isn't sized here, so the policy falls back to its VRAM buckets."""
    diffusion = source if mode == "single_file" else ""
    return ModelFootprint(**reqs.footprint_bytes(diffusion, vae, text))


def _wont_fit_message(fit: FitEstimate) -> str:
    have = f" (you have {fit.total_vram_gb:.0f}GB)" if fit.total_vram_gb else ""
    return (
        f"{fit.note} It needs about {fit.required_vram_gb:.0f}GB of GPU memory{have}. "
        "Use a smaller model or lower the resolution."
    )


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
