"""Assemble the MiniMax H3 modular pipeline from locally placed weights.

Components are constructed here and handed over with ``update_components``, never resolved by class
name: the vendored port defines classes installed diffusers has never heard of, so any name lookup
through diffusers' own registry would fail. ``init_pipeline()`` is called with no repository for the
same reason.

**Unverified against real weights.** Everything up to and including the transformer load is covered
by tests; the component assembly below needs 144 GB on disk and a GPU, so it is written from the
upstream documentation and waits on the numerics gate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

from ...device.policy import DevicePolicy
from ...errors import ComponentError
from .. import pipeline_runtime as rt
from ..offload import (
    apply_offload,
    blocks_that_fit,
    blocks_to_place,
    describe,
    encoder_quantization_config,
    recipe_for,
)
from . import requirements as reqs
from .load import load_transformer

logger = logging.getLogger("inline_core.minimaxh3")

#: From the released model_index.json's `sigma_shift_scales`. Video and audio latents step down two
#: different schedules inside a single transformer call, which is why there are two schedulers.
VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0

#: Left in high precision when quantising. Read off the published unpruned int8 build, which
#: quantises only the attention projections, the FFN and the AdaLN projection.
KEEP_PRECISION = (
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_embedder",
    "time_proj",
    "token_refiner",
    "norm_out",
    "proj_out",
    "audio_proj_out",
)
#: **Conditional on factorisation.** The list above came off the published int8 build, where
#: `adaln_proj.linear` is quantised - but that describes the unfactorised `[96768, 2688]` module,
#: where int8 error averages over 2688 columns. Factorised, the same module is `[96768, 8]` and
#: carries the entire modulation signal through eight columns, so the error concentrates instead of
#: averaging. At 1.5 MB per block and 75 MB across all 50 the saving is not worth that risk, so the
#: factorised projection stays in precision until a measurement says otherwise.
ADALN_KEEP_PRECISION = ("adaln_proj",)

#: The conditioner's own exclusions: its vision tower, embeddings and final norm.
ENCODER_KEEP_PRECISION = (
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
)


def load_pipeline(
    policy: DevicePolicy, *, params: dict[str, Any], partition: str,
    factorise_adaln: bool = True, quantize: bool = True, staged: bool = True,
    transformer: Any = None, video_vae: Any = None, text_encoder: Any = None,
) -> Any:
    """Build, place and cache the pipeline for one partition.

    ``quantize=False`` streams the denoiser in full precision instead of int8, which needs host RAM
    for all 66 GB of it. It exists for the numerics gate, whose whole job is to measure a change
    against weights that nothing else has rounded.

    ``factorise_adaln`` defaults on. It takes the transformer from 66.2 GB to 40.3 GB and cuts 42
    percent off a streamed step, and it was accepted on the numerical measure rather than the pixel
    one: the factorisation perturbs the modulation 14x less than one bf16 ulp of re-rounding, which
    is below the ambiguity the checkpoint's own storage already carries. The gate's pixel tolerance
    could not decide it either way - an information-free re-rounding fails that tolerance too.
    """
    # A wired handle wins over what is on disk. It flows into the cache key through the path, so
    # two graphs pointing at different checkpoints do not share a loaded pipeline.
    transformer_path = _wired(transformer) or _require(
        reqs.resolve_transformer(partition), f"the {partition} transformer"
    )
    encoder_dir = _wired(text_encoder) or _require(
        reqs.resolve("text_encoders", "MiniMax-H3-text-encoder"), "the Qwen3-VL text encoder"
    )
    processor_dir = _require(
        reqs.resolve("text_encoders", "MiniMax-H3-processor"), "the tokenizer and processor"
    )
    video_vae_path = _wired(video_vae) or _require(
        reqs.resolve("vae", reqs.VIDEO_VAE_FILE), "the video VAE"
    )
    audio_vae = _require(reqs.resolve("vae", reqs.AUDIO_VAE_FILE), "the audio VAE")

    # Hand the policy the on-disk sizes so it fits dtype and quantisation to THIS card, then refuse
    # an impossible load up front. Without this it falls back to coarse VRAM buckets and tries to
    # place a 62 GB transformer resident, which no consumer card can hold. Residency is staged, so
    # the encoder is deliberately not counted: it is freed before the denoiser loads.
    from ...device.policy import ModelFootprint

    sizes = reqs.footprint_bytes(partition, factorised=factorise_adaln)
    policy.set_footprint(
        ModelFootprint(diffusion_bytes=sizes["diffusion_bytes"], vae_bytes=sizes["vae_bytes"])
    )
    fit = policy.fit_estimate()
    if fit is not None and not fit.fits:
        raise ComponentError(rt.wont_fit_message(fit))
    _check_host_ram(policy, sizes, fit, quantize=quantize)

    key = rt.PipelineKey(
        arch="minimax-h3",
        diffusion=str(transformer_path),
        vae=str(video_vae_path),
        text_encoder=str(encoder_dir),
        # The two partitions are structurally identical, so without this they would share a cache
        # entry and the second load would silently reuse the first partition's weights.
        variant=partition,
        quant=(policy.quantization().value if quantize else "bf16")
        + ("+adaln" if factorise_adaln else "")
        # Part of the key because it changes where the weights are placed, not just how they run.
        + ("+staged" if staged else ""),
    )
    with rt.PIPELINES.lock:
        cached = rt.PIPELINES.get(key)
        if cached is not None:
            logger.info("MiniMax H3 pipeline cache hit (%s)", partition)
            return cached
        rt.PIPELINES.evict_stale(key)
        pipe = _build(
            policy,
            partition=partition,
            factorise_adaln=factorise_adaln,
            quantize=quantize,
            staged=staged,
            transformer_path=transformer_path,
            encoder_dir=encoder_dir,
            processor_dir=processor_dir,
            video_vae=video_vae_path,
            audio_vae=audio_vae,
        )
        rt.PIPELINES.put(key, pipe)
        return pipe


#: Room left for the process itself, CUDA context, activations and page cache.
_RAM_HEADROOM_GB = 6.0


def _check_host_ram(
    policy: DevicePolicy, sizes: dict[str, int], fit: Any, *, quantize: bool = True
) -> None:
    """Refuse a load that would be killed by the kernel rather than being killed by it.

    ``core/CLAUDE.md`` requires this: a host-RAM OOM does not raise, it takes the whole server down
    with it, so the check has to happen before the first byte is read.

    A quantised plan shrinks each transformer block as it lands, so the peak is the int8 total plus
    the one block being converted, not the 66 GB the file weighs. An unquantised plan really does
    need the whole thing.
    """
    free_mb = policy.free_ram_mb()
    if not free_mb:
        return  # unmeasurable; better to attempt the load than to refuse on no evidence
    free_gb = free_mb / 1024
    resident = sizes["diffusion_bytes"] / 1e9
    if quantize and fit is not None and fit.plan in ("int8", "offload"):
        resident *= 0.55  # int8 weights plus the high-precision layers the exclusion list keeps
    if not quantize:
        # The split-residency planner in `_build` puts whatever will not fit RAM on the card, so the
        # unquantised path is checked against RAM plus VRAM. It still refuses, in `_build`, if the
        # card has no room for the overflow.
        card = (fit.total_vram_gb or 0.0) if fit is not None else 0.0
        resident -= min(resident, max(0.0, card - _VRAM_RESERVE_GB))
    needed = resident + _RAM_HEADROOM_GB
    if free_gb < needed:
        raise ComponentError(
            f"MiniMax H3 needs about {needed:.0f} GB of free system RAM to load, and only "
            f"{free_gb:.0f} GB is free. The transformer is read into RAM in full before it is "
            "quantised, so this is the peak even though the resident size is smaller. Close other "
            "applications, or use a machine with more RAM."
        )


def _wired(ref: Any) -> Path | None:
    """The file behind a wired component handle, or None when the port is unwired."""
    file = getattr(ref, "file", None)
    return Path(file) if file else None


def _require(path: Path | None, what: str) -> Path:
    if path is None:
        raise ComponentError(
            f"MiniMax H3 is missing {what}. Download it from the node's model popup."
        )
    return path


def _build(
    policy: DevicePolicy,
    *,
    partition: str,
    factorise_adaln: bool = False,
    quantize: bool = True,
    staged: bool = False,
    transformer_path: Path,
    encoder_dir: Path,
    processor_dir: Path,
    video_vae: Path,
    audio_vae: Path,
) -> Any:
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    from .vendor import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Blocks,
        MiniMaxH3ModularPipeline,
        MiniMaxH3Ref2VABlocks,
        MiniMaxH3Ref2VAModularPipeline,
        MiniMaxH3Scheduler,
    )

    placement = policy.placement("denoiser")
    dtype = rt.torch_dtype(placement)
    device = "cpu" if placement.offload else str(placement.device)
    # The conditioner is a 32B model that has to be resident alongside, so the denoiser
    # streams: 39 GB of int8 denoiser plus a 19 GB NF4 conditioner exceeds any single card.
    keep = KEEP_PRECISION + (ADALN_KEEP_PRECISION if factorise_adaln else ())
    # `stream_denoiser` asks one question: is the conditioner co-resident during the denoise?
    # Staged, it is not, so the denoiser keeps the card instead of crossing the bus every step.
    recipe = recipe_for(
        policy, keep_precision=keep, stream_denoiser=not staged, quantize=quantize
    )
    # Streaming pins the offloaded weights in host RAM, and pinning 39 GB beside a 21 GB
    # conditioner staging area does not fit a 64 GB machine. Give up the overlap, keep the fit.
    from dataclasses import replace as _replace
    if recipe.use_stream and (policy.free_ram_mb() or 0) / 1024 < 70:
        recipe = _replace(recipe, use_stream=False)
    logger.info("MiniMax H3 memory plan: %s", describe(recipe))

    # The conditioner loads **first**, while host RAM is still free. Its shards stage through
    # ~21 GB of shared memory whatever precision it lands in, and with 39 GB of denoiser
    # already resident there is not enough left for that: the kernel kills the process.
    # NF4, not int8: the conditioner is 66.7 GB in bf16 and runs once per prompt, so it takes the
    # heavier compression while the denoiser keeps int8. That asymmetry is what lets the pair sit
    # beside each other; at int8 on both they need more host RAM than a 64 GB machine has.
    encoder_quant = _encoder_config(recipe)
    _placement_kwargs = _encoder_placement(placement) if encoder_quant is not None else {}
    _encoder_on_card = _placement_kwargs.get("device_map", {}).get("") not in (None, "cpu")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(encoder_dir),
        dtype=dtype,
        local_files_only=True,
        **(
            {"quantization_config": encoder_quant, **_placement_kwargs}
            if encoder_quant is not None
            else {}
        ),
    )

    # The conditioner's shards stage through ~21 GB of shared memory that is not reclaimed on its
    # own, and the denoiser needs that space. Forcing a collection here is what actually frees it.
    _reclaim()

    # With the conditioner placed, whatever VRAM is left decides how much of the denoiser has to
    # live there. Measured now rather than assumed: the conditioner is most of the card.
    resident_blocks = _plan_residency(
        recipe, policy, transformer_path, placement.device,
        # The video VAE is loaded after the denoiser and streams leaf by leaf, which means host RAM.
        later_ram_bytes=video_vae.stat().st_size if recipe.vae_offload else 0,
    )

    # Then the denoiser, into the RAM the conditioner just vacated. Loaded to CPU when the
    # recipe streams: the offload hooks install before placement.
    # Each block is quantised the moment its tensors land, so the full 66 GB bf16 model never
    # exists at once. Quantising afterwards needs that full footprint and gets the process killed.
    transformer = load_transformer(
        transformer_path,
        dtype=dtype,
        # Staged, the denoiser must land in host RAM: the conditioner still holds the card until the
        # prompt is encoded, and 36 GB of int8 weights beside its 19.5 GB fits neither of them.
        # `render_staged` moves it across once the conditioner is parked.
        device="cpu" if (recipe.denoiser_offload or staged) else device,
        shrink=_block_shrinker(
            recipe,
            transformer_path if factorise_adaln else None,
            resident_blocks=resident_blocks,
            resident_device=str(placement.device),
        )
        if (recipe.quantizes or factorise_adaln or resident_blocks)
        else None,
    )
    rt.free_vram()


    # Constructed directly, not via `blocks.init_pipeline()`: that resolves the pipeline class by
    # name out of the installed `diffusers` module, which has never heard of the vendored one and
    # silently falls back to the base `ModularPipeline`. The base class lacks the properties the
    # blocks read for VAE compression ratio, latent channels, sampling rate and patch size, so the
    # failure surfaces much later as a missing attribute part-way into generation.
    blocks = MiniMaxH3Ref2VABlocks() if partition == "ref2va" else MiniMaxH3Blocks()
    pipeline_class = (
        MiniMaxH3Ref2VAModularPipeline if partition == "ref2va" else MiniMaxH3ModularPipeline
    )
    pipe = pipeline_class(blocks=blocks)
    # Ref2VA names its denoiser `transformer_ref`, FL2VA names it `transformer`. `update_components`
    # only *warns* about a name it does not know, so passing the wrong one leaves the pipeline with
    # no denoiser at all and the failure surfaces much later as a missing attribute mid-render.
    denoiser_name = _denoiser_name(blocks)
    pipe.update_components(
        **{denoiser_name: transformer},
        text_encoder=text_encoder,
        tokenizer=AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=True),
        processor=AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True),
        vae=_load_vae(
            # fp32, not the policy's dtype, and not a preference. The vendored keyframe encoder
            # builds its pixels in fp32 on purpose - it says the released model's conditioning
            # cannot be reproduced without that rounding - and unlike the decoder it is not wrapped
            # in autocast, so a bf16 VAE meets fp32 pixels in a conv3d and raises. Text to video
            # never touches that block, which is why only the keyframe nodes broke.
            AutoencoderKLMiniMaxH3, video_vae, dtype=torch.float32, remap=True,
        ),
        # Placed here rather than left to `pipe.to()`: that call is skipped when the denoiser
        # streams, and at 605 MB this one never needs offloading anyway. Without it the decode
        # meets CUDA latents with CPU weights, 20 minutes into a render.
        audio_vae=_load_vae(
            AutoencoderKLMiniMaxH3Audio, audio_vae, dtype=torch.float32, remap="audio",
            device=str(placement.device),
        ),
        scheduler=MiniMaxH3Scheduler(shift=VIDEO_SIGMA_SHIFT),
        audio_scheduler=MiniMaxH3Scheduler(shift=AUDIO_SIGMA_SHIFT),
    )

    # Leaf offload moves the VAE across the bus per leaf, per call. That is the right trade when
    # the card is full, and badly wrong when it is not: at 960x544 it turned a 6 minute render into
    # 32. Staged, the conditioner is on the CPU by decode time, so if the measured free VRAM covers
    # the VAE plus a working margin it stays resident instead.
    vae_resident = staged and _vae_fits(video_vae, placement.device)
    if vae_resident:
        recipe = _replace(recipe, vae_offload=None)

    apply_offload(
        recipe,
        resident_blocks=resident_blocks,
        denoiser=transformer,
        # Skipped only when `from_pretrained` already put it on the card. Left on the CPU because
        # it did not fit, it needs the hooks like anything else.
        encoder=None if _encoder_on_card else getattr(text_encoder, "model", text_encoder),
        vae=getattr(pipe, "vae", None),
        device=placement.device,
    )
    if vae_resident:
        pipe.vae.to(rt.torch_device(placement))
    pipe._inline_denoiser = denoiser_name  # noqa: SLF001 - our own attribute on our own pipeline
    # Whether the phase swap may hand the denoiser the whole card. False on the offload plan, where
    # it carries streaming hooks and is far larger than the card: moving it wholesale is an OOM.
    pipe._inline_resident_denoiser = recipe.denoiser_offload is None  # noqa: SLF001
    if staged:
        # Split once, at build time: the halves share every component object, so this costs nothing
        # but the block graph. `render_staged` below does the swapping.
        pipe._inline_phases = tuple(  # noqa: SLF001 - our own attribute on our own pipeline
            pipeline_class(blocks=half)
            for half in rt.split_blocks(blocks, through="text_encoder")
        )
        for half in pipe._inline_phases:  # noqa: SLF001
            half.update_components(**dict(pipe.components))
        transformer.to("cpu")  # the conditioner has the card until the prompt is encoded
        rt.free_vram()
    elif not recipe.denoiser_offload:
        pipe.to(device)
    elif resident_blocks:
        # The streamed tail carries its own hooks; everything else - the head blocks are already
        # there - is placed once so no forward meets a CPU weight beside a CUDA activation.
        _place_unstreamed(transformer, resident_blocks, placement.device)
    return pipe


#: Left free beside a resident VAE: the denoiser is already placed, so this only has to cover the
#: decode's own activations, which are the largest single allocation of the render.
_VAE_RESIDENT_MARGIN_GB = 12.0


def _vae_fits(path: Path, device: Any) -> bool:
    """Whether the video VAE can stay on the card rather than streaming leaf by leaf.

    fp32 doubles what the file weighs, and that is what actually has to fit.
    """
    free = rt.free_vram_bytes(device)
    if not free:
        return False
    needed = path.stat().st_size * 2 + int(_VAE_RESIDENT_MARGIN_GB * 1e9)
    fits = free > needed
    logger.info(
        "MiniMax H3 video VAE: %s (%.1f GB free, needs %.1f GB with margin)",
        "resident" if fits else "streamed leaf by leaf", free / 1e9, needed / 1e9,
    )
    return fits


def _denoiser_name(blocks: Any) -> str:
    """Whichever name this blockset gives its denoiser. Declared, not guessed by partition."""
    declared = {getattr(spec, "name", "") for spec in (blocks.expected_components or [])}
    for candidate in ("transformer", "transformer_ref"):
        if candidate in declared:
            return candidate
    raise ComponentError("This MiniMax H3 blockset declares no denoiser component.")


def render_staged(pipe: Any, device: Any, **call: Any) -> Any:
    """Encode the prompt, get the conditioner off the card, then denoise with the card to itself.

    H3's conditioner is a 32B model that rivals its denoiser, and holding both means the denoiser
    streams every block of every step. Encoding is one pass per prompt, so the two never actually
    need the card at the same time - but the convenience call runs all eight blocks together, which
    is what makes it look as though they do.

    Falls back to the single call when the pipeline was not built staged, so a caller never has to
    ask which kind it holds.
    """
    phases = getattr(pipe, "_inline_phases", None)
    if phases is None:
        return pipe(**call)
    head, tail = phases
    device = str(device)

    # Every kwarg goes to BOTH halves, not just the first. `set_timesteps` lives in the tail, so a
    # head-only handoff left it reading the descriptor default: a render that asked for 8 steps
    # silently took 49. The halves ignore what their own blocks do not declare, so this is safe.
    state = head(**call)
    # Parked, not released: the next render needs it again, and 19.5 GB across the bus is seconds
    # against the minutes of streaming it buys back.
    denoiser = getattr(pipe, getattr(pipe, "_inline_denoiser", "transformer"))
    resident = getattr(pipe, "_inline_resident_denoiser", True)

    # Parking the conditioner is worth doing either way: it frees the card for activations even when
    # the denoiser is too big to take it. Moving the denoiser across is only right when it fits.
    pipe.text_encoder.to("cpu")
    rt.free_vram()
    if resident:
        denoiser.to(device)
    try:
        return tail(state, **call)
    finally:
        if resident:
            denoiser.to("cpu")
        pipe.text_encoder.to(device)
        rt.free_vram()


def _reclaim() -> None:
    """Drop staging buffers between two loads that cannot both fit."""
    import gc

    gc.collect()
    rt.free_vram()


def _encoder_placement(placement: Any) -> dict[str, Any]:
    """Where the conditioner goes.

    Pinned to the card when it fits, which is the fast path every large card takes. When it does not
    - the 4-bit conditioner is about 20 GB - it goes to host RAM instead and the offload recipe's
    leaf hooks stream it a module at a time. Splitting it across both was tried and is a dead end:
    bitsandbytes will not dispatch a 4-bit model to CPU without keeping that share in fp32, and
    accelerate's own dispatch then trips over parameters left on meta.

    The conditioner runs once per prompt, so streaming it is a far better trade than it would be for
    the denoiser, which is read every step.
    """
    free = rt.free_vram_bytes(placement.device)
    needed = int(_ENCODER_RESIDENT_GB * 1e9)
    if free and free < needed:
        logger.info(
            "MiniMax H3 conditioner: streamed from host RAM (%.1f GB free, wants %.1f GB)",
            free / 1e9, needed / 1e9,
        )
        return {"device_map": {"": "cpu"}}
    return {"device_map": {"": str(placement.device)}}


#: What the 4-bit conditioner occupies once placed, measured on an L40S.
_ENCODER_RESIDENT_GB = 20.5


def _encoder_config(recipe: Any) -> Any:
    """The conditioner's quantisation, decided independently of the denoiser's.

    Its own exclusion list, and its own rung: at 66.7 GB in bf16 this conditioner is larger than the
    denoiser it conditions and larger than any card this node targets, so 4-bit is not a step on a
    ladder here, it is the only way it loads at all. A caller running the denoiser in full precision
    still gets a 4-bit conditioner, and since both sides of a comparison share it, it tilts neither.
    """
    from dataclasses import replace

    from ...device.policy import Quantization

    return encoder_quantization_config(
        replace(recipe, quantize=Quantization.INT8, keep_precision=ENCODER_KEEP_PRECISION)
    )


#: Left free on the card beside any resident blocks: activations, the streamed block in flight, and
#: fragmentation. The measured peak at 608x352 was 5.8 GB above the resident conditioner, so this is
#: that with margin. It only caps the split, which moves the host overflow and never more, and the
#: full-precision path it guards is barely fitting at any canvas.
_VRAM_RESERVE_GB = 10.0

#: Host RAM the split leaves free, and deliberately much larger than `_RAM_HEADROOM_GB`. That one
#: covers a load; this one covers the whole render running on top of 50 GB of weights that never
#: leave RAM. Sized off a failure: at 6 GB the planner left 1.9 GB free at peak, and with no swap
#: and the page cache already down to 745 MB there was nothing for the kernel to reclaim, so the
#: box reset rather than killing anything. Aggregate capacity was never the problem, the split was.
_SPLIT_HEADROOM_GB = 12.0


def _plan_residency(
    recipe: Any, policy: DevicePolicy, source: Path, device: Any, *, later_ram_bytes: int = 0
) -> int:
    """How many leading blocks to place on the card so the rest fits host RAM.

    Zero for a quantised load, which fits RAM with room. The full-precision path is the case this
    exists for: 66 GB of weights against a 64 GB machine has nowhere to sit, and the difference
    between placing the overflow and letting the kernel swap it is minutes per step.

    ``later_ram_bytes`` is what components loaded *after* the denoiser will claim from the same RAM.
    Free memory read here is not free memory at peak, and the difference is not small: H3's video
    VAE is leaf-offloaded, so its 5.2 GB lands in host RAM after this decision is made.

    A second correction is folded into ``_SPLIT_HEADROOM_GB`` rather than measured, because it
    cannot be: while the model streams in, its CPU-side weights are clean file-backed pages from the
    safetensors mmap, which the kernel can drop and re-read for nothing. The first denoising step
    ends that. Group offload returns each block with ``module.to("cpu")``, which allocates fresh
    anonymous memory and drops the file-backed storage, so a footprint that was reclaimable at load
    is unreclaimable a step later. Anything sizing this from ``available`` is reading a number that
    is about to stop being true.
    """
    if recipe.quantizes or not recipe.denoiser_offload:
        return 0
    block_bytes = _block_bytes(source)
    # `free_ram_mb` is GB-times-1024, not mebibytes, which is how `_check_host_ram` reads it too.
    free_ram = int((policy.free_ram_mb() or 0) / 1024 * 1e9)
    if not block_bytes or not free_ram:
        return 0
    model_bytes = source.stat().st_size
    needed = blocks_to_place(
        model_bytes=model_bytes,
        block_bytes=block_bytes,
        free_ram_bytes=free_ram,
        ram_headroom_bytes=int(_SPLIT_HEADROOM_GB * 1e9) + later_ram_bytes,
    )
    if not needed:
        return 0
    affordable = blocks_that_fit(
        free_vram_bytes=rt.free_vram_bytes(device),
        block_bytes=block_bytes,
        reserve_bytes=int(_VRAM_RESERVE_GB * 1e9),
    )
    if needed > affordable:
        raise ComponentError(
            f"MiniMax H3 in full precision needs {needed} of its blocks on the accelerator to fit "
            f"{free_ram / 1e9:.0f} GB of system RAM, and there is room for {affordable}. Run it "
            "quantised, or use a machine with more RAM."
        )
    # The projected figure is logged because it is the number that decides whether the machine
    # survives the render, and it is not observable anywhere else until it is too late.
    logger.info(
        "MiniMax H3 split residency: %d of %d blocks placed on %s (%.1f GB), %.1f GB stays in RAM, "
        "leaving %.1f GB free",
        needed, _num_blocks(source), device, needed * block_bytes / 1e9,
        (model_bytes - needed * block_bytes) / 1e9,
        (free_ram - (model_bytes - needed * block_bytes)) / 1e9,
    )
    return needed


def _place_unstreamed(transformer: Any, resident: int, device: Any) -> None:
    """Place everything the offload hooks are not responsible for: the head blocks and the
    embedders, norms and projections around them."""
    from ..offload import block_stack

    # Core's `Device` is not torch's, and `Module.to` takes no interest in the difference.
    device = str(device)
    stack = block_stack(transformer)
    for child in transformer.children():
        if child is not stack:
            child.to(device)
    for block in list(stack)[:resident]:
        block.to(device)
    for tensor in (
        *transformer.parameters(recurse=False),
        *transformer.buffers(recurse=False),
    ):
        tensor.data = tensor.data.to(device)


def _header(path: Path) -> dict[str, Any]:
    """A safetensors header, without reading a byte of the tensors."""
    import json
    import struct

    try:
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            parsed = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _block_bytes(path: Path) -> int:
    """What one transformer block weighs on disk, summed from the header's byte offsets."""
    total = 0
    for key, info in _header(path).items():
        if key.startswith("blocks.0.") and isinstance(info, dict):
            start, end = info.get("data_offsets", (0, 0))
            total += end - start
    return total


def _num_blocks(path: Path) -> int:
    return len({k.split(".")[1] for k in _header(path) if k.startswith("blocks.") and "." in k[7:]})


def _load_vae(
    cls: Any, path: Path, *, dtype: torch.dtype, remap: str | bool = False,
    device: Any = None,
) -> Any:
    """A VAE from a single consolidated file.

    The published files carry their source config in the safetensors metadata, so nothing else
    has to be shipped beside them. ``remap`` runs the video VAE through its key plan: like the
    transformer, it is written for MiniMax's implementation, with a CompVis-spelled encoder, a fused
    per-head-interleaved attention and a gated FFN whose halves are the other way round.
    """
    import inspect

    from safetensors.torch import load_file

    # The embedded metadata describes the publisher's own implementation, so it carries keys the
    # diffusers port has no argument for (`source_config`, `vae_clip_length`, `sample_rate`, ...).
    # Passing them through is a TypeError, so only what the constructor accepts is kept.
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    config = {k: v for k, v in _metadata_config(path).items() if k in accepted}
    model = cls(**config) if config else cls()
    state = load_file(str(path))
    if remap:
        targets = sorted(dict(model.named_parameters()) | dict(model.named_buffers()))
        state = _remapped_vae_state(state, audio=(remap == "audio"), target_keys=targets)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unfilled = [key for key in missing if not _self_computed(key)]
    if unfilled:
        raise ComponentError(
            f"{path.name} is missing {len(unfilled)} tensors this VAE needs, starting with "
            f"{unfilled[0]}. It is probably a different build than this node expects."
        )
    if unexpected:
        logger.warning("%s carries %d tensors the VAE does not use", path.name, len(unexpected))
    model = model.to(dtype=dtype) if device is None else model.to(dtype=dtype, device=device)
    return model.eval()


def _self_computed(key: str) -> bool:
    """Rotary tables the port derives from its geometry, as the DiT does with `rope.inv_freq`."""
    return bool(re.search(r"(^|\.)(rope|freqs|inv_freq)", key))


def _remapped_vae_state(
    state: dict[str, Any], *, audio: bool = False, target_keys: list[str] | None = None
) -> dict[str, Any]:
    from ..keymap import transform
    from . import vae_keys

    plan = (
        vae_keys.build_audio_plan(sorted(state), target_keys or [])
        if audio
        else vae_keys.build_plan(sorted(state))
    )
    out: dict[str, Any] = {}
    for key, tensor in state.items():
        # The layout check needs whole-tensor statistics, and these are already validated by the
        # coverage check plus the shapes the port declares.
        for target, value in transform(key, tensor, plan.actions[key], verify_layout=False):
            out[target] = value
    return out


def _metadata_config(path: Path) -> dict[str, Any]:
    """The VAE config the publisher embedded in the file's safetensors metadata, if any."""
    import json
    import struct

    try:
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return {}
    meta = header.get("__metadata__") or {}
    for key, raw in meta.items():
        if "vae" in key and isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return {k: v for k, v in parsed.items() if not k.startswith("_")}
    return {}


def _adaln_basis(source: Path) -> Any:
    """The low-rank basis for the AdaLN branch, read straight from the checkpoint.

    Derived before the model streams, because the shrink callback needs it while the first block
    lands and ``time_embedder.*`` sorts after ``blocks.*``. Only the two timestep-embedding tensors
    are read, about 60 MB of a 66 GB file.
    """
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from safetensors import safe_open

    from . import adaln as adaln_mod

    with safe_open(str(source), framework="pt") as handle:
        get = handle.get_tensor
        proj_in = get("time_embedder.proj_in.weight")
        proj_out = get("time_embedder.proj_out.weight")
        embedder = TimestepEmbedding(
            in_channels=proj_in.shape[1], time_embed_dim=proj_in.shape[0], out_dim=proj_out.shape[0]
        )
        embedder.linear_1.weight.data = proj_in.float()
        embedder.linear_1.bias.data = get("time_embedder.proj_in.bias").float()
        embedder.linear_2.weight.data = proj_out.float()
        embedder.linear_2.bias.data = get("time_embedder.proj_out.bias").float()
    proj = Timesteps(num_channels=proj_in.shape[1], flip_sin_to_cos=True, downscale_freq_shift=0)
    return adaln_mod.derive_basis(proj, embedder)


def _block_shrinker(
    recipe: Any,
    factorise_source: Path | None = None,
    *,
    resident_blocks: int = 0,
    resident_device: str = "",
) -> Any:
    """Shrink one transformer block as soon as its weights land, and place it if it is a resident.

    Order matters and is the rule in ``models/offload.py``: the structural transform runs on
    the unquantised weights, and quantisation is last. Reversed, the factorisation would try to
    matrix-multiply an ``Int8Tensor`` and raise inside torchao. Placement comes after both, and
    happens here rather than after the load for the same reason the shrinking does: a block moved to
    the card as it lands is a block that never has to fit host RAM alongside the other forty-nine.
    """
    basis = _adaln_basis(factorise_source) if factorise_source is not None else None
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError:
        if factorise_source is None:
            logger.warning("torchao is absent; MiniMax H3 stays in full precision.")
            return None
        Int8WeightOnlyConfig = quantize_ = None  # type: ignore[assignment]

    def keep(module: Any, name: str) -> bool:
        # Linear only: torchao asserts the module has a `weight`, so a norm or a container reaching
        # here is an AssertionError rather than a skip. The exclusion list then removes the layers
        # the published int8 build left in high precision.
        if not isinstance(module, torch.nn.Linear):
            return False
        return not any(part in name for part in recipe.keep_precision)

    def shrink(model: Any, prefix: str) -> None:
        from . import adaln as adaln_mod

        module = model
        for part in prefix.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        module.requires_grad_(False)
        if basis is not None:            # clause 1: structural transform, unquantised weights
            module.adaln_proj = adaln_mod.factorise_block(module, basis)
        if recipe.quantizes:             # clause 2: quantisation last
            quantize_(module, Int8WeightOnlyConfig(version=2), filter_fn=keep)
        index = prefix.rsplit(".", 1)[-1]
        if resident_blocks and index.isdigit() and int(index) < resident_blocks:
            module.to(resident_device)

    return shrink
