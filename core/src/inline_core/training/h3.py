"""MiniMax H3's precache, which has to run in two passes that never overlap.

Every other arch loads its VAE and its text encoder together and encodes both in one sweep. H3
cannot: the video VAE is ~10 GB resident in fp32 and the Qwen3-VL conditioner is ~19 GB even at
4-bit, so together they are most of a 46 GB card before a single latent exists. Pixels are encoded
and the VAE dropped, then captions are encoded and the conditioner dropped, and only then does the
62 GB transformer load.

The layout of H3's packed sequence depends only on the caption length and the latent grid, so it is
built here once per image rather than per step.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("inline_core.training.h3")

#: H3's (t, h, w) patch, and the transformer's ``audio_in_channels``. A still has no audio rows, but
#: the empty tensor still has to be the width ``audio_proj_in`` expects.
PATCH = (1, 2, 2)
AUDIO_LATENT_CHANNELS = 32

#: What an absent caption becomes. H3 tokenises with ``add_special_tokens=False`` and has no BOS,
#: so the empty string is genuinely zero tokens and the conditioner reshapes a (1, 0) sequence into
#: an attention head and raises. A single space is one real token and the closest thing this model
#: has to no caption: it is guidance-distilled, so inference never encodes an unconditional prompt
#: and there is no established unconditional embedding to match. Covers caption dropout and an
#: image whose ``.txt`` is missing or blank.
_EMPTY_CAPTION = " "


def precache(
    dataset_dir: str,
    models_dir: str,
    device: str,
    dtype: Any,
    resolution: int,
    flip: bool,
    want_unconditional: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Every image as a latent and every caption as conditioning, as CPU tensors."""
    from . import dataset as ds

    pairs = ds._pairs(Path(dataset_dir))
    if not pairs:
        raise RuntimeError("The exported dataset is empty.")

    root = Path(models_dir)
    latents = _encode_pixels(root, pairs, device, resolution, flip)
    captions = [caption for _img, caption in pairs for _ in ((False, True) if flip else (False,))]
    if want_unconditional:
        captions.append("")
    embeds = _encode_captions(root, captions, device, dtype)

    items = [
        {"latent": latent, **_conditioning(embed, tags, latent)}
        for latent, (embed, tags) in zip(latents, embeds, strict=False)
    ]
    unconditional = None
    if want_unconditional:
        embed, tags = embeds[-1]
        # Caption dropout swaps in a different text length, which moves every row after it, so the
        # whole layout travels with the embedding rather than just the embedding.
        unconditional = _conditioning(embed, tags, latents[0])
    return items, unconditional


def _encode_pixels(
    root: Path, pairs: list[tuple[Path, str]], device: str, resolution: int, flip: bool
) -> list[Any]:
    """Pass one: the video VAE, then dropped."""
    import torch
    from PIL import Image

    from ..models.minimaxh3.vendor.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD

    vae = _load_video_vae(root, device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

    from . import dataset as ds

    out: list[Any] = []
    try:
        for img_path, _caption in pairs:
            for mirrored in (False, True) if flip else (False,):
                square = ds._square(Image.open(img_path), resolution, mirrored)
                # H3 normalises with ImageNet statistics, not to [-1, 1] like the image archs, and
                # a still is one frame: (1, 3, 1, H, W).
                raw = torch.from_numpy(_as_array(square)).to(device)
                pixels = raw.permute(2, 0, 1)[None, :, None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                with torch.no_grad():
                    # The spatial encoder alone, the path inference uses for a single frame; the
                    # temporal chunking is for 17n+5 clips.
                    latent = _sample(vae._encode_clip(pixels))
                out.append(((latent.cpu() - mean) / std)[0])
    finally:
        del vae
        _reclaim()
    logger.info("MiniMax H3: cached %d latents, video VAE released", len(out))
    return out


def _encode_captions(
    root: Path, captions: list[str], device: str, dtype: Any
) -> list[tuple[Any, Any]]:
    """Pass two: the 4-bit conditioner, then dropped."""
    import torch

    from ..models.minimaxh3.vendor.encoders import MiniMaxH3TextEncoderStep

    pipeline = _load_conditioner(root, device, dtype)
    out: list[tuple[Any, Any]] = []
    try:
        for caption in captions:
            caption = caption or _EMPTY_CAPTION
            with torch.no_grad():
                # The staticmethod rather than the block, so nothing needs a PipelineState. `dtype`
                # is not optional here: it defaults to `components.transformer.dtype`, and this
                # pipeline deliberately has no transformer.
                embeds, tags = MiniMaxH3TextEncoderStep.encode_prompt(
                    pipeline, caption, None, device=torch.device(device), dtype=dtype
                )
            out.append((embeds[0].cpu(), tags.cpu()))
    finally:
        _drop_conditioner(pipeline)
    logger.info("MiniMax H3: cached %d captions, conditioner released", len(out))
    return out


def _conditioning(embed: Any, tags: Any, latent: Any) -> dict[str, Any]:
    """The packed layout for one caption over one still, as cacheable tensors."""
    import torch

    from ..models.minimaxh3.vendor.packing import build_packed_sequence, build_row_timesteps

    _channels, frames, height, width = latent.shape
    layout = build_packed_sequence(
        text_token_tags=tags,
        num_latent_frames=frames,
        latent_height=height,
        latent_width=width,
        num_audio_latents=0,
        patch_size=PATCH,
        keyframe_anchors=(),
    )
    # Training pins no conditioning rows and a still has no audio rows, so every row shares one
    # noise level and this vector is constant across steps. Derived from the vendored planner and
    # then checked, so a change upstream is caught here rather than silently mis-addressing the
    # AdaLN table.
    unique, indices = build_row_timesteps(layout, 1.0, 1.0, 1.0, 1.0)
    if unique.numel() != 1 or bool(indices.any()):
        raise RuntimeError(
            "MiniMax H3's row-timestep plan is no longer single-valued for a still; the cached "
            "timestep index vector would mis-address the AdaLN table."
        )
    return {
        "embed": embed,
        "audio": torch.zeros(0, AUDIO_LATENT_CHANNELS),
        "timestep_indices": indices,
        "token_tags": layout.token_tags,
        "position_ids": layout.position_ids,
        "video_indices": layout.video_indices,
        "audio_indices": layout.audio_indices,
        "text_indices": layout.text_indices,
    }


def _load_video_vae(root: Path, device: str) -> Any:
    import torch

    from ..models.minimaxh3 import requirements as reqs
    from ..models.minimaxh3.pipeline import _load_vae
    from ..models.minimaxh3.vendor import AutoencoderKLMiniMaxH3
    from . import models

    path = Path(models._require(root, "minimax-h3", "vae"))
    # fp32, not the compute dtype: the encoder builds its pixels in fp32 and a half-precision VAE
    # meets them in a conv3d and raises. Same reason the generation pipeline pins it.
    del reqs
    return _load_vae(
        AutoencoderKLMiniMaxH3, path, dtype=torch.float32, remap=True, device=device
    )


def _load_conditioner(root: Path, device: str, dtype: Any) -> Any:
    """The Qwen3-VL conditioner at 4-bit, on a transformer-less pipeline.

    Routed through the vendored pipeline rather than a local copy of the prompt template, for the
    same reason Krea 2 and FLUX.2 build a transformer-less pipeline: the tokenisation, the vision
    token type ids and the layer-50 tap are all easy to get subtly wrong and impossible to notice.
    """
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    from ..models.minimaxh3 import requirements as reqs
    from ..models.minimaxh3.vendor import MiniMaxH3ModularPipeline
    from ..models.minimaxh3.vendor.encoders import MiniMaxH3TextEncoderStep
    from . import models

    encoder_dir = Path(models._require(root, "minimax-h3", "text_encoders"))
    processor_dir = reqs.resolve("text_encoders", "MiniMax-H3-processor")
    if processor_dir is None:
        raise RuntimeError(
            "MiniMax H3's tokenizer and processor are missing. Download them from the node's model "
            "popup; the conditioner cannot tokenise a caption without them."
        )

    quant, placement = _conditioner_plan(device)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(encoder_dir),
        dtype=dtype,
        local_files_only=True,
        **({"quantization_config": quant, **placement} if quant is not None else {}),
    )
    pipeline = MiniMaxH3ModularPipeline(blocks=MiniMaxH3TextEncoderStep())
    pipeline.update_components(
        text_encoder=text_encoder,
        tokenizer=AutoTokenizer.from_pretrained(str(processor_dir), local_files_only=True),
        processor=AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True),
    )
    return pipeline


def _conditioner_plan(device: str) -> tuple[Any, dict[str, Any]]:
    """4-bit, and on the card only if it fits, reusing the generation path's own decision."""
    from ..device.memory import MemoryPolicy
    from ..models.minimaxh3.pipeline import _encoder_config, _encoder_placement
    from ..models.offload import recipe_for

    policy = MemoryPolicy()
    placement = policy.placement("text_encoder")
    recipe = recipe_for(policy, keep_precision=(), stream_denoiser=False, quantize=True)
    quant = _encoder_config(recipe)
    if quant is None:  # bitsandbytes absent: nothing to place, and the load will speak for itself
        return None, {}
    del device
    return quant, _encoder_placement(placement)


def _drop_conditioner(pipeline: Any) -> None:
    for name in ("text_encoder", "tokenizer", "processor"):
        try:
            setattr(pipeline, name, None)
        except (AttributeError, TypeError):  # a ModularPipeline may guard its component slots
            pass
    _reclaim()


def load_base(models_dir: str, device: str, dtype: Any, quant: Any) -> Any:
    """The frozen base: streamed a block at a time, factorised, then quantised.

    Order is the rule in ``models/offload.py`` and it is not negotiable: the structural transform
    runs on unquantised weights, quantisation is last. Reversed, ``factorise_block`` would try to
    matrix-multiply a 4-bit blob whose ``.weight`` is a uint8 column, and the result would be
    garbage rather than an error.

    Nothing here holds the whole model: at bf16 it is 62 GB on disk and each block is shrunk as its
    weights land, so the peak is the shrunk total plus the one block being converted.
    """
    from ..device.policy import Quantization
    from ..models.minimaxh3 import load as h3_load
    from ..models.minimaxh3.pipeline import _adaln_basis
    from . import models

    path = Path(models._require(Path(models_dir), "minimax-h3", "diffusion_models"))
    # Derived before the stream: the callback needs it while the first block lands, and
    # `time_embedder.*` sorts after `blocks.*`. Only two tensors are read, about 60 MB of 62 GB.
    basis = _adaln_basis(path)
    staging = "cpu" if quant is Quantization.NF4 else device
    model = h3_load.load_transformer(
        path, dtype=dtype, device=staging, shrink=_shrinker(basis, quant, device, dtype)
    )
    _place_unstreamed(model, device)
    logger.info("MiniMax H3 base loaded for training (%s)", quant.value)
    return model


def _shrinker(basis: Any, quant: Any, device: str, dtype: Any) -> Any:
    """Shrink one transformer block as soon as its weights land, and place it."""
    from ..device.policy import Quantization
    from ..models.minimaxh3 import adaln as adaln_mod

    def shrink(model: Any, prefix: str) -> None:
        module = model
        for part in prefix.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        module.requires_grad_(False)
        module.adaln_proj = adaln_mod.factorise_block(module, basis)  # clause 1: unquantised
        if quant is Quantization.NF4:  # clause 2: quantisation last
            _swap_to_4bit(module, keep=_keeps_precision)
            module.to(device)  # bitsandbytes quantizes each weight during this move
        else:
            module.to(device=device, dtype=dtype)

    return shrink


def _keeps_precision(path: str) -> bool:
    """Whether a Linear is spared the 4-bit swap.

    Only the factorised AdaLN projection. Unfactorised it is [96768, 2688] and quantising it is
    ordinary; factorised it is [96768, 8] and those eight columns carry the entire modulation
    signal, so the error concentrates instead of averaging. It is 1.5 MB a block, 75 MB across the
    stack, which is not worth that.
    """
    return "adaln_proj" in path


def _swap_to_4bit(module: Any, keep: Any = None, prefix: str = "") -> None:
    """Replace every ``nn.Linear`` under ``module`` with a bitsandbytes NF4 layer.

    The QLoRA arrangement: the base is frozen and 4-bit, gradients still flow through it to the
    adapter on top, and quantization itself happens when the layer moves to CUDA.

    Deliberately not ``loaders._swap_to_4bit``, which takes no keep-predicate and would convert the
    factorised AdaLN projection along with everything else. Kept local rather than widening the
    shared loader, so the two shipping architectures that use it are untouched.
    """
    import bitsandbytes as bnb
    import torch

    for name, child in list(module.named_children()):
        path = f"{prefix}{name}"
        if isinstance(child, torch.nn.Linear) and not (keep is not None and keep(path)):
            layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
                quant_type="nf4",
            )
            layer.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type="nf4"
            )
            if child.bias is not None:
                layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, layer)
        else:
            _swap_to_4bit(child, keep, f"{path}.")


def _place_unstreamed(model: Any, device: str) -> None:
    """Move what the block callback never sees.

    ``load.py`` fires ``shrink`` only for ``transformer_blocks.N``, so the embedders, the token
    refiner, the norms and the two output heads are still wherever the stream staged them. Left on
    the CPU they meet CUDA activations in the first forward.
    """
    stack = model.transformer_blocks
    for child in model.children():
        if child is not stack:
            child.to(device)
    for tensor in (*model.parameters(recurse=False), *model.buffers(recurse=False)):
        tensor.data = tensor.data.to(device)


def _sample(moments: Any) -> Any:
    """Sample the posterior. Unlike the conditioning path this takes no fixed seed and no fp16
    round trip: those exist to make a *reference* reproducible, and baking them into training
    latents would narrow what the LoRA ever sees."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    return DiagonalGaussianDistribution(moments).sample()


def _as_array(image: Any) -> Any:
    import numpy as np

    return np.asarray(image, dtype="uint8")


def _reclaim() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
