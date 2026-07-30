"""Prompt embeddings for FLUX.2, cached on disk so iterating never reloads the text encoder.

FLUX.2's encoder is a large fraction of the model: Qwen3-4B is roughly half of klein 4B's total
footprint, and dev's Mistral-3 dwarfs even the 32B transformer. The encoder is also the one
component whose output depends on nothing but the prompt, so re-encoding it to change a seed or a
resolution is pure waste - and on a tight card it is the difference between the denoise fitting and
not.

So the embedding is computed once, written to the engine data dir keyed by everything that can
change it, and read back on later runs. A cache hit skips the encoder entirely: the pipeline is
built with the encoder detached and the denoise gets its VRAM. Roughly 8 MB per klein prompt and
16 MB per dev prompt, pruned oldest-first against a size cap.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from ...config import data_dir
from ...device.policy import DevicePolicy
from .. import pipeline_runtime as rt

logger = logging.getLogger("inline_core.flux2")

#: Total bytes of cached embeddings to keep. Small next to a checkpoint, and one entry per distinct
#: prompt, so this holds hundreds of prompts before anything is evicted.
_CACHE_LIMIT_BYTES = 2 * 1024**3


def _root() -> Path:
    return data_dir() / "embeds" / "flux2"


def cache_key(
    *, prompt: str, text_encoder_file: str, layers: tuple[int, ...], max_sequence_length: int,
    dtype: str,
) -> str:
    """Everything that changes the embedding. The encoder *file* is in the key, not just the
    variant, so swapping in a different Qwen3 build invalidates rather than silently reusing."""
    parts = (
        prompt, text_encoder_file, ",".join(str(n) for n in layers), str(max_sequence_length), dtype
    )
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


def load(key: str) -> Any | None:
    """A cached embedding tensor, or None. A corrupt or unreadable entry just misses."""
    path = _root() / f"{key}.pt"
    if not path.is_file():
        return None
    try:
        import torch

        embeds = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a bad cache entry must never break a run
        logger.warning("Discarding unreadable prompt-embed cache entry %s", path.name)
        path.unlink(missing_ok=True)
        return None
    path.touch()  # keep recently used entries away from the pruner
    return embeds


def store(key: str, embeds: Any) -> None:
    """Write an embedding to the cache. Best-effort - a full or read-only disk is not an error."""
    try:
        import torch

        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{key}.pt"
        staging = path.with_suffix(".part")
        torch.save(rt.embeds_to(embeds, "cpu"), staging)
        staging.replace(path)  # atomic, so a crash mid-write never leaves a half entry
        _prune(root)
    except Exception as error:  # noqa: BLE001 - caching is an optimization, never a requirement
        logger.debug("Could not cache prompt embeddings: %s", error)


def _prune(root: Path) -> None:
    entries = sorted(
        (p for p in root.glob("*.pt") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True
    )
    total = 0
    for entry in entries:
        total += entry.stat().st_size
        if total > _CACHE_LIMIT_BYTES:
            entry.unlink(missing_ok=True)


def prompt_kwargs(
    pipe: Any,
    policy: DevicePolicy,
    *,
    prompt: str,
    negative: str | None,
    text_encoder_file: str,
    layers: tuple[int, ...],
    max_sequence_length: int = 512,
) -> dict[str, Any]:
    """Pipeline call kwargs carrying precomputed embeddings, from cache when possible.

    Falls back to handing the pipeline the raw prompt if anything goes wrong, so this optimization
    can never be the reason a render fails.
    """
    device = str(policy.placement("denoiser").device)
    dtype = policy.placement("text_encoder").dtype.value
    key = cache_key(
        prompt=prompt,
        text_encoder_file=text_encoder_file,
        layers=layers,
        max_sequence_length=max_sequence_length,
        dtype=dtype,
    )
    negative_key = (
        None
        if negative is None
        else cache_key(
            prompt=negative,
            text_encoder_file=text_encoder_file,
            layers=layers,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
        )
    )

    cached = load(key)
    cached_negative = load(negative_key) if negative_key else None
    if cached is not None and (negative_key is None or cached_negative is not None):
        logger.info("Prompt-embed cache hit - skipping the text encoder entirely")
        kwargs: dict[str, Any] = {"prompt_embeds": rt.embeds_to(cached, device)}
        if cached_negative is not None:
            kwargs["negative_prompt_embeds"] = rt.embeds_to(cached_negative, device)
        return kwargs

    def raw() -> dict[str, Any]:
        return {"prompt": prompt}

    def encode(target: str) -> dict[str, Any]:
        # encode_prompt is not wrapped in the pipeline's @torch.no_grad (only __call__ is); the
        # caller (encoded_prompt_kwargs) supplies it.
        import torch

        embeds, _ids = pipe.encode_prompt(
            prompt=prompt,
            device=torch.device(target),
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=layers,
        )
        store(key, embeds)
        out: dict[str, Any] = {"prompt_embeds": rt.embeds_to(embeds, target)}
        if negative is not None and negative_key is not None:
            negative_embeds, _negative_ids = pipe.encode_prompt(
                prompt=negative,
                device=torch.device(target),
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=layers,
            )
            store(negative_key, negative_embeds)
            out["negative_prompt_embeds"] = rt.embeds_to(negative_embeds, target)
        return out

    return rt.encoded_prompt_kwargs(pipe, policy, encode=encode, fallback=raw)
