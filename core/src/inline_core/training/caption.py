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

#: BLIP ships *inside* transformers (no `trust_remote_code`), so it survives runtime upgrades.
#: Florence-2 was the original default but its remote code breaks on current transformers
#: (`Florence2LanguageConfig has no attribute forced_bos_token_id`); it still works through
#: ``INLINE_CAPTIONER_MODEL`` if you pin a compatible transformers - see `_caption_one`.
_DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"
_TASK = "<DETAILED_CAPTION>"


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


def _caption_one(model: Any, processor: Any, device: str, path: str) -> str:
    """One caption. Handles both shapes: task-token models (Florence-2, which post-processes a
    tagged string) and plain image-captioning models (BLIP), which just decode the output."""
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
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

    model_id = os.environ.get("INLINE_CAPTIONER_MODEL") or _DEFAULT_MODEL
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
