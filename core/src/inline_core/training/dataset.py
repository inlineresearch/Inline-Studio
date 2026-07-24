"""Dataset precache: encode every training image to a VAE latent and every caption to text
embeddings **once**, up front, so the VAE + text encoder can be freed and the loop only touches the
transformer + LoRA (the big low-VRAM win).

The orchestrator exports the dataset as ``NNNN.<img>`` + ``NNNN.txt`` (caption) pairs. Images are
center-cropped to a square ``resolution``; aspect-ratio bucketing is a follow-up. Training
conditioning must mirror inference exactly or the LoRA learns against an embedding the generation
path never produces, so each arch reuses its own pipeline's encoder rather than a copy of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import arch as archs

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
    import numpy as np
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
    arr = np.asarray(img, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW


def precache(
    dataset_dir: str,
    components: Any,
    arch: str,
    device: str,
    dtype: Any,
    resolution: int,
) -> list[dict[str, Any]]:
    """Return one cached item per image/caption pair, as CPU tensors."""
    import torch
    from PIL import Image

    pairs = _pairs(Path(dataset_dir))
    if not pairs:
        raise RuntimeError("The exported dataset is empty.")

    encode_latent = _krea2_latent if arch == archs.KREA2 else _zimage_latent
    encode_caption = _krea2_caption if arch == archs.KREA2 else _zimage_caption
    cached: list[dict[str, Any]] = []
    for img_path, caption in pairs:
        pixels = _to_tensor(Image.open(img_path), resolution).to(device, dtype).unsqueeze(0)
        with torch.no_grad():
            latent = encode_latent(components.vae, pixels)
            item = encode_caption(components, caption, device)
        item["latent"] = latent.squeeze(0).cpu()
        cached.append({k: v.cpu() for k, v in item.items()})
    return cached


# --- Z-Image ------------------------------------------------------------------------------------


def _zimage_latent(vae: Any, pixels: Any) -> Any:
    # Model-space latent: decode is (z / scaling) + shift, so the forward normalization is
    # (z - shift) * scaling. Dropping the shift skews every training latent off-manifold.
    scaling = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return (vae.encode(pixels).latent_dist.sample() - shift) * scaling


def _zimage_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    """Caption -> conditioning, matching ``ZImagePipeline._encode_prompt`` exactly: the Qwen chat
    template (thinking on), the PENULTIMATE hidden layer, padding stripped."""
    tokenizer, text_encoder = components.tokenizer, components.text_encoder
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
    return {"embed": hidden[0][mask[0]]}  # (real_seq_len, dim), padding removed


# --- Krea 2 -------------------------------------------------------------------------------------


def _krea2_latent(vae: Any, pixels: Any) -> Any:
    """The Qwen-Image VAE is a video codec, so pixels carry a length-1 temporal axis, and latents
    are normalized per channel with ``latents_mean``/``latents_std``, not a scalar scale."""
    import torch

    latent = vae.encode(pixels.unsqueeze(2)).latent_dist.sample()
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype)
    std = torch.tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype)
    return ((latent - mean.view(shape)) / std.view(shape)).squeeze(2)


def _krea2_caption(components: Any, caption: str, device: str) -> dict[str, Any]:
    """Caption -> a (seq, 12, 2560) stack of tapped hidden layers + its mask, straight from
    ``Krea2Pipeline.encode_prompt`` so the prompt template and layer taps cannot drift from
    inference. Padding is kept: the mask carries it, and the rotary positions depend on it."""
    import torch

    embeds, mask = components.pipeline.encode_prompt(
        prompt=caption or "", device=torch.device(device)
    )
    return {"embed": embeds[0], "mask": mask[0]}
