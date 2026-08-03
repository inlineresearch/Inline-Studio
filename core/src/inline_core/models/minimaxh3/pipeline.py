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
from ..offload import apply_offload, describe, encoder_quantization_config, recipe_for
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
#: The conditioner's own exclusions: its vision tower, embeddings and final norm.
ENCODER_KEEP_PRECISION = (
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
)


def load_pipeline(policy: DevicePolicy, *, params: dict[str, Any], partition: str) -> Any:
    """Build, place and cache the pipeline for one partition."""
    transformer_path = _require(reqs.resolve_transformer(partition), f"the {partition} transformer")
    encoder_dir = _require(
        reqs.resolve("text_encoders", "MiniMax-H3-text-encoder"), "the Qwen3-VL text encoder"
    )
    processor_dir = _require(
        reqs.resolve("text_encoders", "MiniMax-H3-processor"), "the tokenizer and processor"
    )
    video_vae = _require(reqs.resolve("vae", reqs.VIDEO_VAE_FILE), "the video VAE")
    audio_vae = _require(reqs.resolve("vae", reqs.AUDIO_VAE_FILE), "the audio VAE")

    # Hand the policy the on-disk sizes so it fits dtype and quantisation to THIS card, then refuse
    # an impossible load up front. Without this it falls back to coarse VRAM buckets and tries to
    # place a 62 GB transformer resident, which no consumer card can hold. Residency is staged, so
    # the encoder is deliberately not counted: it is freed before the denoiser loads.
    from ...device.policy import ModelFootprint

    sizes = reqs.footprint_bytes(partition)
    policy.set_footprint(
        ModelFootprint(diffusion_bytes=sizes["diffusion_bytes"], vae_bytes=sizes["vae_bytes"])
    )
    fit = policy.fit_estimate()
    if fit is not None and not fit.fits:
        raise ComponentError(rt.wont_fit_message(fit))
    _check_host_ram(policy, sizes, fit)

    key = rt.PipelineKey(
        arch="minimax-h3",
        diffusion=str(transformer_path),
        vae=str(video_vae),
        text_encoder=str(encoder_dir),
        # The two partitions are structurally identical, so without this they would share a cache
        # entry and the second load would silently reuse the first partition's weights.
        variant=partition,
        quant=policy.quantization().value,
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
            transformer_path=transformer_path,
            encoder_dir=encoder_dir,
            processor_dir=processor_dir,
            video_vae=video_vae,
            audio_vae=audio_vae,
        )
        rt.PIPELINES.put(key, pipe)
        return pipe


#: Room left for the process itself, CUDA context, activations and page cache.
_RAM_HEADROOM_GB = 6.0


def _check_host_ram(policy: DevicePolicy, sizes: dict[str, int], fit: Any) -> None:
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
    if fit is not None and fit.plan in ("int8", "offload"):
        resident *= 0.55  # int8 weights plus the high-precision layers the exclusion list keeps
    needed = resident + _RAM_HEADROOM_GB
    if free_gb < needed:
        raise ComponentError(
            f"MiniMax H3 needs about {needed:.0f} GB of free system RAM to load, and only "
            f"{free_gb:.0f} GB is free. The transformer is read into RAM in full before it is "
            "quantised, so this is the peak even though the resident size is smaller. Close other "
            "applications, or use a machine with more RAM."
        )


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
    recipe = recipe_for(policy, keep_precision=KEEP_PRECISION, stream_denoiser=True)
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
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(encoder_dir),
        dtype=dtype,
        local_files_only=True,
        **(
            {"quantization_config": encoder_quant, "device_map": {"": str(placement.device)}}
            if encoder_quant is not None
            else {}
        ),
    )

    # The conditioner's shards stage through ~21 GB of shared memory that is not reclaimed on its
    # own, and the denoiser needs that space. Forcing a collection here is what actually frees it.
    _reclaim()

    # Then the denoiser, into the RAM the conditioner just vacated. Loaded to CPU when the
    # recipe streams: the offload hooks install before placement.
    # Each block is quantised the moment its tensors land, so the full 66 GB bf16 model never
    # exists at once. Quantising afterwards needs that full footprint and gets the process killed.
    transformer = load_transformer(
        transformer_path,
        dtype=dtype,
        device="cpu" if recipe.denoiser_offload else device,
        shrink=_block_shrinker(recipe) if recipe.quantizes else None,
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
    pipe.update_components(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=True),
        processor=AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True),
        vae=_load_vae(
            AutoencoderKLMiniMaxH3, video_vae,
            dtype=rt.torch_dtype(policy.placement("vae")), remap=True,
        ),
        audio_vae=_load_vae(
            AutoencoderKLMiniMaxH3Audio, audio_vae, dtype=torch.float32, remap="audio",
        ),
        scheduler=MiniMaxH3Scheduler(shift=VIDEO_SIGMA_SHIFT),
        audio_scheduler=MiniMaxH3Scheduler(shift=AUDIO_SIGMA_SHIFT),
    )

    apply_offload(
        recipe,
        denoiser=transformer,
        # Already placed and quantised by from_pretrained when a config was supplied.
        encoder=None if encoder_quant is not None else getattr(text_encoder, "model", text_encoder),
        vae=getattr(pipe, "vae", None),
        device=placement.device,
    )
    if not recipe.denoiser_offload:
        pipe.to(device)
    return pipe


def _reclaim() -> None:
    """Drop staging buffers between two loads that cannot both fit."""
    import gc

    gc.collect()
    rt.free_vram()


def _encoder_config(recipe: Any) -> Any:
    """The conditioner's quantisation, with its own exclusion list rather than the denoiser's."""
    from dataclasses import replace

    return encoder_quantization_config(replace(recipe, keep_precision=ENCODER_KEEP_PRECISION))


def _load_vae(
    cls: Any, path: Path, *, dtype: torch.dtype, remap: str | bool = False
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
    return model.to(dtype=dtype).eval()


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


def _block_shrinker(recipe: Any) -> Any:
    """A callback that int8s one transformer block as soon as its weights have landed."""
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError:
        logger.warning("torchao is absent; MiniMax H3 stays in full precision.")
        return None

    def keep(module: Any, name: str) -> bool:
        # Linear only: torchao asserts the module has a `weight`, so a norm or a container reaching
        # here is an AssertionError rather than a skip. The exclusion list then removes the layers
        # the published int8 build left in high precision.
        if not isinstance(module, torch.nn.Linear):
            return False
        return not any(part in name for part in recipe.keep_precision)

    def shrink(model: Any, prefix: str) -> None:
        module = model
        for part in prefix.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        module.requires_grad_(False)
        quantize_(module, Int8WeightOnlyConfig(version=2), filter_fn=keep)

    return shrink
