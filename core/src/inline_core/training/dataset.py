"""Dataset precache: encode every training image to a VAE latent and every caption to text
embeddings **once**, up front, so the VAE + text encoder can be freed and the loop only touches the
transformer + LoRA (the big low-VRAM win the plan calls for).

The orchestrator exports the dataset as ``NNNN.<img>`` + ``NNNN.txt`` (caption) pairs. For the MVP
we resize to a square ``resolution`` (center-crop); aspect-ratio bucketing (ai-toolkit
``buckets.py``) is a follow-up. NOTE (needs GPU + weights): the exact conditioning tensor Z-Image
expects (hidden states vs. pooled) must be validated against the Z-Image runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _pairs(dataset_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for img in sorted(dataset_dir.iterdir()):
        if img.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        caption_file = img.with_suffix(".txt")
        caption = caption_file.read_text(encoding="utf-8").strip() if caption_file.exists() else ""
        out.append((img, caption))
    return out


def _to_tensor(image: Any, resolution: int) -> Any:
    import torch
    from PIL import Image

    img = image.convert("RGB")
    # Center-crop to square, then resize to the training resolution; pixels normalized to [-1, 1].
    short = min(img.size)
    left = (img.width - short) // 2
    top = (img.height - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW


def precache(
    dataset_dir: str,
    components: Any,
    device: str,
    dtype: Any,
    resolution: int,
) -> list[dict[str, Any]]:
    """Return ``[{"latent", "embed"}]`` (CPU tensors) for every image/caption pair."""
    import torch
    from PIL import Image

    pairs = _pairs(Path(dataset_dir))
    if not pairs:
        raise RuntimeError("The exported dataset is empty.")

    vae, text_encoder, tokenizer = (
        components.vae,
        components.text_encoder,
        components.tokenizer,
    )
    scaling = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    cached: list[dict[str, Any]] = []
    for img_path, caption in pairs:
        pixels = _to_tensor(Image.open(img_path), resolution).to(device, dtype).unsqueeze(0)
        with torch.no_grad():
            latent = vae.encode(pixels).latent_dist.sample() * scaling
            tokens = tokenizer(
                caption or "", return_tensors="pt", padding="max_length",
                truncation=True, max_length=256,
            ).to(device)
            embed = text_encoder(**tokens).last_hidden_state
        cached.append({"latent": latent.squeeze(0).cpu(), "embed": embed.squeeze(0).cpu()})
    return cached
