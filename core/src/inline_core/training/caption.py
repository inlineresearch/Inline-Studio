"""Local auto-captioner: ``python -m inline_core.training.caption`` (reads a manifest on stdin).

The orchestrator (``studio/training.py`` ``auto_caption``) pipes ``{"items": [{"id", "path"}]}`` in
and reads ``{"id", "caption"}`` JSON lines back. Runs as a subprocess so transformers/torch never
import server-side. Uses BLIP by default; ``INLINE_CAPTIONER_MODEL`` overrides it. The weights are
fetched once into the HF cache (opt-in, only when the user hits "Auto-caption"), the same
fetch-once posture the Z-Image runner uses for its reference components.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from . import protocol

#: The captioners the UI offers, first is the default. **Add a new caption model here**: give it a
#: stable ``id``, a human ``label``, and its Hugging Face ``repo``. BLIP ships inside transformers
#: (no ``trust_remote_code``) so it is the safe default and survives runtime upgrades. Anything on
#: the Hub that ``AutoProcessor`` + an image-text-to-text (or remote-code causal-LM) class can load
#: works too; weights download once into the HF cache on first use. Florence-2 is reachable but its
#: remote code needs a pinned transformers (see the ``_load`` fallback), so it is not listed here by
#: default - add it if you have pinned a compatible transformers.
CAPTIONERS: list[dict[str, str]] = [
    {
        "id": "blip-large",
        "label": "BLIP large (default)",
        "repo": "Salesforce/blip-image-captioning-large",
    },
    {
        "id": "blip-base",
        "label": "BLIP base (faster, lighter)",
        "repo": "Salesforce/blip-image-captioning-base",
    },
]
_DEFAULT_MODEL = CAPTIONERS[0]["repo"]
_TASK = "<DETAILED_CAPTION>"


def available_captioners() -> list[dict[str, str]]:
    """The captioner list for the UI. Torch-free, so ``studio`` can serve it without importing the
    ML stack."""
    return [dict(c) for c in CAPTIONERS]


def _resolve_model(manifest: dict[str, Any]) -> str:
    """Which captioner to run: the UI's choice (a curated id or a raw HF repo) wins, then the
    ``INLINE_CAPTIONER_MODEL`` override, then the default."""
    chosen = str(manifest.get("model") or "").strip()
    if chosen:
        for c in CAPTIONERS:
            if c["id"] == chosen:
                return c["repo"]
        return chosen  # a raw HF repo id passed straight through
    return os.environ.get("INLINE_CAPTIONER_MODEL") or _DEFAULT_MODEL


def _load(model_id: str, token: Any = None) -> tuple[Any, Any, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True, "token": token}
    # The in-transformers captioners (BLIP et al): `AutoModelForVision2Seq` was renamed
    # `AutoModelForImageTextToText` in transformers 5, so resolve whichever this build has.
    image_text = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
        transformers, "AutoModelForVision2Seq", None
    )
    try:
        if image_text is None:
            raise ValueError("no image-text auto class")
        model = image_text.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        # Florence-2 and friends register themselves as causal LMs via their remote code.
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    return model, processor, device


def _looks_like_auth_failure(error: Exception) -> bool:
    text = str(error).lower()
    return "401" in text or "not a valid model identifier" in text or "repository not found" in text


def _load_with_fallback(model_id: str) -> tuple[Any, Any, Any]:
    """Load the captioner, retrying anonymously if a stored HF token is rejected.

    A stale/expired token in the HF cache makes the hub return 401 for *public* repos too, which
    surfaces as a confusing "not a valid model identifier". The captioner is public, so an
    anonymous retry recovers instead of failing the whole run."""
    try:
        return _load(model_id)
    except Exception as first:  # noqa: BLE001 - retried anonymously below, then re-raised
        if not _looks_like_auth_failure(first):
            raise
        try:
            return _load(model_id, token=False)
        except Exception:  # noqa: BLE001 - report the original failure, it's the informative one
            raise first from None


def _open(path: str) -> Any:
    """The frame to caption. A clip is captioned from its middle frame, which is more
    representative than the first and stops PIL raising on a container it cannot read."""
    from pathlib import Path

    from PIL import Image

    from . import dataset as ds

    if not ds.is_video(Path(path)):
        return Image.open(path).convert("RGB")

    from ..models.minimaxh3.vendor.packing_ref2va import decode_reference_video

    frames, _fps, _audio = decode_reference_video(path)
    return Image.fromarray(frames[len(frames) // 2]).convert("RGB")


def _caption_one(model: Any, processor: Any, device: str, path: str) -> str:
    """One caption. Handles both shapes: task-token models (Florence-2, which post-processes a
    tagged string) and plain image-captioning models (BLIP), which just decode the output."""
    import torch

    image = _open(path)
    task_style = hasattr(processor, "post_process_generation")
    inputs = (
        processor(text=_TASK, images=image, return_tensors="pt")
        if task_style
        else processor(images=image, return_tensors="pt")
    ).to(device, model.dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128, num_beams=3, do_sample=False)
    if not task_style:
        return str(processor.decode(ids[0], skip_special_tokens=True)).strip()
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=_TASK, image_size=(image.width, image.height)
    )
    return str(parsed.get(_TASK, "")).strip()


def main() -> int:
    try:
        manifest = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        protocol.error(f"Bad caption manifest: {exc}")
        return 2
    items = manifest.get("items") or []
    if not items:
        return 0

    model_id = _resolve_model(manifest)
    try:
        model, processor, device = _load_with_fallback(model_id)
    except Exception as exc:  # noqa: BLE001 - a missing captioner degrades to no captions
        protocol.error(f"Captioner unavailable: {exc}")
        return 1

    for item in items:
        try:
            caption = _caption_one(model, processor, device, item["path"])
        except Exception as exc:  # noqa: BLE001 - one bad image shouldn't sink the batch
            protocol.error(f"Caption failed for {item.get('id')}: {exc}")
            continue
        protocol.emit({"id": item["id"], "caption": caption})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
