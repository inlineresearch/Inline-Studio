"""Resolve + load the Z-Image components for training, reusing the inference loaders.

Bring-your-own-weights, same as generation: the base transformer / VAE / text encoder are the single
files the user already dropped under ``models/`` for Z-Image. Training additionally needs, in Turbo
mode, the **training/assistant adapter** that undoes turbo distillation during training - we apply
it by fusing it into the base with the existing LoRA fuser (``models/lora.py``), so the base behaves
de-distilled while the trainable LoRA learns on top.

NOTE (needs GPU + weights to finalize): the assistant-adapter semantics and the LoRA target modules
for ``ZImageTransformer2DModel`` should be validated against the ai-toolkit reference
(``toolkit/assistant_lora.py``, ``extensions_built_in/diffusion_models/z_image/*``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ARCH = "z-image"
_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".sft")


@dataclass(frozen=True)
class Components:
    transformer: Any
    vae: Any
    text_encoder: Any
    tokenizer: Any
    scheduler: Any


def _first_weight(root: Path) -> str | None:
    if not root.is_dir():
        return None
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
    )
    return str(files[0]) if files else None


def _require(root: Path, category: str, env: str) -> str:
    picked = os.environ.get(env) or _first_weight(root / category)
    if not picked:
        raise RuntimeError(
            f"No {category} weight found under {root / category}. Add the Z-Image {category} file "
            f"there (or set {env})."
        )
    return picked


def _adapter_path(root: Path, base_mode: str) -> str | None:
    if base_mode != "turbo_adapter":
        return None
    picked = os.environ.get("INLINE_ZIMAGE_TRAIN_ADAPTER")
    if not picked:
        loras = root / "loras"
        if loras.is_dir():
            picked = next(
                (str(p) for p in sorted(loras.iterdir()) if "adapter" in p.name.lower()), None
            )
    if not picked:
        raise RuntimeError(
            "Turbo mode needs a training adapter to avoid turbo drift. Add the Z-Image training "
            "adapter to models/loras/ (or set INLINE_ZIMAGE_TRAIN_ADAPTER), or use de-turbo mode."
        )
    return picked


def compute_dtype() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_components(models_dir: str, base_mode: str, device: str, dtype: Any) -> Components:
    """Load the base transformer (grad-enabled, unquantized), VAE, text encoder + tokenizer, and the
    flow-match scheduler. In Turbo mode the training adapter is fused into the transformer first."""
    from ..graph.loader_runners import LoraRef
    from ..models import loaders

    root = Path(models_dir)
    diffusion = _require(root, "diffusion_models", "INLINE_ZIMAGE_MODEL")
    vae_file = _require(root, "vae", "INLINE_ZIMAGE_VAE")
    encoder_file = _require(root, "text_encoders", "INLINE_ZIMAGE_TEXT_ENCODER")

    adapter = _adapter_path(root, base_mode)
    loras: tuple[LoraRef, ...] = (LoraRef(file=adapter, strength=1.0),) if adapter else ()

    transformer = loaders.load_diffusion(_ARCH, diffusion, dtype, device=device, loras=loras)
    vae = loaders.load_vae(_ARCH, vae_file, dtype, device=device)
    text_encoder, tokenizer = loaders.load_text_encoder(_ARCH, encoder_file, dtype, device=device)
    scheduler = loaders.load_scheduler(_ARCH)
    return Components(transformer, vae, text_encoder, tokenizer, scheduler)
