"""Local auto-captioner: ``python -m inline_core.training.caption`` (reads a manifest on stdin).

The orchestrator (``studio/training.py`` ``auto_caption``) pipes ``{"items": [{"id", "path"}]}`` in
and reads ``{"id", "caption"}`` JSON lines back. Runs as a subprocess so transformers/torch never
import server-side. Uses Florence-2 (small, ~0.23B) by default; ``INLINE_CAPTIONER_MODEL`` overrides
it. The weights are fetched once into the HF cache (opt-in, only when the user hits "Auto-caption"),
the same fetch-once posture the Z-Image runner uses for its reference components.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from . import protocol

_DEFAULT_MODEL = "microsoft/Florence-2-base"
_TASK = "<DETAILED_CAPTION>"


def _load(model_id: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device


def _caption_one(model: Any, processor: Any, device: str, path: str) -> str:
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    inputs = processor(text=_TASK, images=image, return_tensors="pt").to(device, model.dtype)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
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
        model, processor, device = _load(model_id)
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
