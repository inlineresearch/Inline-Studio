"""Dataset precache: encode every training image to a VAE latent and every caption to text
embeddings **once**, up front, so the VAE + text encoder can be freed and the loop only touches the
transformer + LoRA (the big low-VRAM win the plan calls for).

The orchestrator exports the dataset as ``NNNN.<img>`` + ``NNNN.txt`` (caption) pairs. For the MVP
we resize to a square ``resolution`` (center-crop); aspect-ratio bucketing (ai-toolkit
``buckets.py``) is a follow-up. The conditioning tensor mirrors ``ZImagePipeline._encode_prompt``
exactly (chat template + penultimate hidden layer + padding stripped) - see ``_encode_caption``.
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
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    cached: list[dict[str, Any]] = []
    for img_path, caption in pairs:
        pixels = _to_tensor(Image.open(img_path), resolution).to(device, dtype).unsqueeze(0)
        with torch.no_grad():
            # Model-space latent: decode is (z / scaling) + shift, so the forward normalization is
            # (z - shift) * scaling. Dropping the shift skews every training latent off-manifold.
            latent = (vae.encode(pixels).latent_dist.sample() - shift) * scaling
            embed = _encode_caption(text_encoder, tokenizer, caption, device)
        cached.append({"latent": latent.squeeze(0).cpu(), "embed": embed.cpu()})
    return cached


def _encode_caption(text_encoder: Any, tokenizer: Any, caption: str, device: str) -> Any:
    """Caption -> conditioning, matching ``ZImagePipeline._encode_prompt`` exactly: wrap in the Qwen
    chat template (thinking on), take the PENULTIMATE hidden layer (not the last), and strip padding
    to the real tokens (variable-length (seq, dim)). Training conditioning must mirror inference or
    the LoRA learns against an embedding the generation path never produces."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": caption or ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(
        [text], padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    mask = inputs.attention_mask.bool()
    hidden = text_encoder(
        input_ids=inputs.input_ids, attention_mask=mask, output_hidden_states=True
    ).hidden_states[-2]
    return hidden[0][mask[0]]  # (real_seq_len, dim), padding removed
