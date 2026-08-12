"""Build the LTX-2.5 training cache: every clip as a latent, every caption as conditioning.

Two passes, because the video VAE and Gemma cannot usefully share a card: the VAE encodes every clip
and is freed, then the encoder runs over every caption and is freed, and only then does the
transformer load into an empty card. `h3.py` stages the same way for the same reason.

Decoding is **trimmed, not sampled**. A fixed window means one VAE encode per clip, which is the
whole point of a precache; re-cutting a different window every step would defeat caching the latents
at all. The grid only snaps down, so part of every clip is always dropped - ``clip_window="end"``
takes the tail instead of the head, for footage whose action is at the finish.

Motion mode encodes a second clip per item as the reference. It is held as a
``VideoConditionByReferenceLatent`` at ``downscale_factor=1``: upstream allows a smaller reference
than the target, but the factor used in training has to match the one used at inference, and a
mismatch degrades output without raising. Keeping it at 1 removes the chance to get that wrong.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import arch as archs

logger = logging.getLogger(__name__)

#: LTX's video VAE compresses 8 pixel frames into 1 latent frame plus a head, and 32 pixels into 1
#: latent pixel. Mirrored here only to size the reported latent, never to reshape one.
_TEMPORAL_FACTOR = 8
_SPATIAL_FACTOR = 32

#: How many captions go through Gemma at once. Its activations scale with the batch, and it is a
#: 12B model sharing the card with nothing else at this point - but six at once still OOMed 44 GiB.
_CAPTION_BATCH = 1

#: A reference at the target's own resolution. Upstream supports downscaling for cheaper training,
#: but the factor has to match at inference and a mismatch is silent - see the module docstring.
_REFERENCE_DOWNSCALE = 1


class ShortClipError(RuntimeError):
    """A clip too short for the grid. Skipped rather than fatal: one bad file in a folder of
    hundreds should not end the run."""


def precache(
    dataset_dir: str,
    models_dir: str,
    device: str,
    dtype: Any,
    resolution: int,
    flip: bool,
    want_unconditional: bool,
    clip_frames: int = 1,
    clip_window: str = "start",
    mode: str = archs.MODE_CLIP,
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Every clip as a latent and every caption as conditioning, as CPU tensors."""
    from . import dataset as ds

    say = on_status or (lambda _text: None)
    triples = ds.media_triples(Path(dataset_dir))
    if not triples:
        raise RuntimeError("The exported dataset is empty.")
    if mode == archs.MODE_MOTION:
        unpaired = [t.target.name for t in triples if t.reference is None]
        if unpaired:
            raise RuntimeError(
                f"A Motion LoRA needs a reference clip for every item; {len(unpaired)} have none "
                f"(first: {unpaired[0]}). Pair them in the dataset panel, or train a Clip LoRA."
            )

    grid = archs.ARCHS[archs.LTX25].clip
    assert grid is not None, "the LTX arch declares a clip grid"
    frames = grid.snap(max(clip_frames, grid.min_frames))

    latents, references, kept = _encode_clips(
        Path(models_dir), triples, device, dtype, resolution, frames, clip_window, mode, say
    )
    if not kept:
        raise RuntimeError(
            f"None of the {len(triples)} dataset items could be encoded. Each clip must be at "
            f"least {frames} frames at {grid.fps:g}fps ({frames / grid.fps:.2f}s)."
        )

    captions = [t.caption for t in kept]
    embeds, unconditional = _encode_captions(
        Path(models_dir), captions, device, dtype, want_unconditional, say
    )

    items: list[dict[str, Any]] = []
    for index, (latent, embed) in enumerate(zip(latents, embeds, strict=True)):
        item: dict[str, Any] = {"latent": latent, "embed": embed}
        if mode == archs.MODE_MOTION:
            item["reference"] = references[index]
        items.append(item)
    say(f"cached {len(items)} items")
    return items, unconditional


def _encode_clips(
    root: Path,
    triples: list[Any],
    device: str,
    dtype: Any,
    resolution: int,
    frames: int,
    clip_window: str,
    mode: str,
    say: Callable[[str], None],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Every clip through the video VAE, in one build-and-free of the encoder.

    Only the clips that survived encoding are returned, because a caption list that still holds a
    skipped clip's entry pairs every later caption with the wrong latent.
    """
    import torch

    from ..models.ltx25 import requirements as reqs
    from ..models.ltx25.vendor.ltx_pipelines.utils.blocks import ImageConditioner

    vae = reqs.resolve("vae", reqs.VIDEO_VAE_FILE)
    if vae is None:
        raise RuntimeError(
            "LTX-2.5 training needs the video VAE. Download it from an LTX node's model popup."
        )

    latents: list[Any] = []
    references: list[Any] = []
    kept: list[Any] = []

    def run(encoder: Any) -> None:
        for index, triple in enumerate(triples):
            say(f"encoding clip {index + 1} of {len(triples)}")
            try:
                pixels = _clip_pixels(
                    triple.target, resolution, frames, clip_window, device, dtype
                )
            except ShortClipError as exc:
                logger.warning("%s", exc)
                continue
            with torch.no_grad():
                latent = encoder(pixels)
            if mode == archs.MODE_MOTION:
                ref_pixels = _clip_pixels(
                    triple.reference, resolution, frames, clip_window, device, dtype
                )
                with torch.no_grad():
                    references.append(_reference_condition(encoder(ref_pixels).to("cpu")))
            latents.append(latent[0].to("cpu"))
            kept.append(triple)

    say(f"loading the video VAE ({_vram()})")
    ImageConditioner(str(vae), dtype, torch.device(device))(run)
    # The encoder's context frees the model, but the allocator keeps its blocks - and Gemma is the
    # next thing onto this card. Measured: without this the text pass OOMs with 44.35 GiB in use.
    _reclaim()
    say(f"encoded {len(kept)} of {len(triples)} clips ({_vram()})")
    return latents, references, kept


def _reclaim() -> None:
    """Hand the card back between passes. The VAE and Gemma are never both needed."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram() -> str:
    """What this process holds, and what the whole card has left.

    Both halves, and the device-wide one is the important half. ``memory_allocated`` counts only
    this process: it read 0.0 GiB through three OOM diagnoses while a separate Inline Core server
    held 15 GiB of the same card, which is a very convincing way to look at an empty number and
    conclude the card is empty.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    gib = 1024**3
    free, total = torch.cuda.mem_get_info()
    return (
        f"{torch.cuda.memory_allocated() / gib:.1f} GiB ours, "
        f"{free / gib:.1f} of {total / gib:.1f} GiB free on the card"
    )


def _check_card(needed_gib: float, say: Callable[[str], None]) -> None:
    """Warn when another process has taken the room this pass needs.

    A precache that dies inside an RMSNorm tells the user nothing about the generation server they
    left running in another window. Naming it costs one call to the driver.
    """
    import torch

    if not torch.cuda.is_available():
        return
    gib = 1024**3
    free, total = torch.cuda.mem_get_info()
    used_elsewhere = (total - free) / gib - torch.cuda.memory_allocated() / gib
    if free / gib < needed_gib and used_elsewhere > 1.0:
        say(
            f"warning: {used_elsewhere:.1f} GiB of this card is held by another process and only "
            f"{free / gib:.1f} GiB is free. Stop any running Inline Core server before training."
        )


def _clip_pixels(
    path: Path, resolution: int, frames: int, window: str, device: str, dtype: Any
) -> Any:
    """A clip as ``(1, 3, frames, resolution, resolution)`` in the VAE's [-1, 1] range.

    Resampled onto the model's frame rate before trimming, so "48 frames" means the same duration
    whatever the source was shot at.
    """
    import torch

    from ..models.ltx25.vendor.ltx_pipelines.utils.media_io import (
        decode_video_from_file,
        video_preprocess,
    )

    grid = archs.ARCHS[archs.LTX25].clip
    assert grid is not None
    decoded = list(decode_video_from_file(str(path), device))
    if not decoded:
        raise ShortClipError(f"{path.name} decoded to no frames. Skipped.")

    resampled = _resample(decoded, _source_fps(path), grid.fps)
    if len(resampled) < frames:
        raise ShortClipError(
            f"{path.name} is {len(resampled)} frames once resampled to {grid.fps:g}fps, below the "
            f"{frames} this run needs ({frames / grid.fps:.2f}s). Skipped."
        )
    kept = resampled[-frames:] if window == "end" else resampled[:frames]
    return video_preprocess(iter(kept), resolution, resolution, dtype, torch.device(device))


def _source_fps(path: Path) -> float:
    """The clip's own frame rate, or the model's when the container does not say."""
    import av

    grid = archs.ARCHS[archs.LTX25].clip
    assert grid is not None
    try:
        with av.open(str(path)) as container:
            rate = container.streams.video[0].average_rate
            return float(rate) if rate else grid.fps
    except Exception:  # noqa: BLE001 - an unreadable rate is not worth failing the whole run over
        return grid.fps


def _resample(frames: list[Any], source_fps: float, target_fps: float) -> list[Any]:
    """Pick frames so the clip plays at ``target_fps`` without changing its duration.

    Nearest-neighbour in time rather than blended: a blended frame is not a frame the camera ever
    saw, and the VAE is being asked what real footage looks like.
    """
    if source_fps <= 0 or abs(source_fps - target_fps) < 1e-3:
        return frames
    duration = len(frames) / source_fps
    wanted = int(duration * target_fps)
    if wanted <= 0:
        return []
    step = len(frames) / wanted
    return [frames[min(len(frames) - 1, int(i * step))] for i in range(wanted)]


def _reference_condition(latent: Any) -> Any:
    """A reference latent as the conditioning item the transformer forward applies."""
    from ..models.ltx25.vendor.ltx_core.conditioning.types.reference_video_cond import (
        VideoConditionByReferenceLatent,
    )

    return VideoConditionByReferenceLatent(
        reference_latent=latent, downscale_factor=_REFERENCE_DOWNSCALE
    )


def _encode_captions(
    root: Path,
    captions: list[str],
    device: str,
    dtype: Any,
    want_unconditional: bool,
    say: Callable[[str], None],
) -> tuple[list[Any], dict[str, Any] | None]:
    """Every caption through Gemma, in one build-and-free of the encoder.

    The empty prompt is encoded in the same pass when caption dropout is on, rather than reloading
    24 GiB of encoder later for one string.
    """
    import torch

    from ..models.ltx25 import requirements as reqs
    from ..models.ltx25.vendor.ltx_pipelines.utils.blocks import PromptEncoder
    from ..models.ltx25.vendor.ltx_pipelines.utils.model_paths import ModelPaths

    transformer = reqs.resolve_transformer("dev")
    encoder = reqs.resolve("text_encoders", reqs.TEXT_ENCODER_FILE)
    if transformer is None or encoder is None:
        raise RuntimeError(
            "LTX-2.5 training needs the dev transformer and the Gemma 4 text encoder. Download "
            "them from an LTX node's model popup."
        )

    say(f"encoding {len(captions)} captions ({_vram()})")
    # Gemma alone measured 21.9 GiB on an L40S, encoding one caption at a time.
    _check_card(22.0, say)
    prompts = [*captions, ""] if want_unconditional else list(captions)
    from ..models.ltx25.vendor.ltx_pipelines.utils.types import OffloadMode

    prompt_encoder = PromptEncoder(
        ModelPaths.from_split(
            transformer_path=str(transformer), text_encoder_path=str(encoder)
        ),
        dtype,
        torch.device(device),
        # Streamed, not resident. Gemma is 24.5 GiB and is wanted once per run; holding it whole
        # here is what put 44.35 GiB on a 44.39 GiB card and OOMed the caption pass.
        offload_mode=OffloadMode.CPU,
    )
    # Encoded in small batches, not one call. Gemma is a 12B model and its activations scale with
    # the batch: six captions at once OOMed a 44 GiB card, and a real dataset is hundreds. The
    # encoder is built once and reused across batches, so this costs nothing but peak memory.
    embeds: list[Any] = []
    for start in range(0, len(prompts), _CAPTION_BATCH):
        batch = prompts[start : start + _CAPTION_BATCH]
        say(f"encoding captions {start + 1}-{start + len(batch)} of {len(prompts)} ({_vram()})")
        embeds += [c.video_encoding[0].to("cpu") for c in prompt_encoder(batch)]
    _reclaim()

    if want_unconditional:
        return embeds[:-1], {"embed": embeds[-1]}
    return embeds, None
