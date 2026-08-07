"""Resolve + load the components for training, reusing the inference loaders.

Bring-your-own-weights, same as generation: the base transformer / VAE / text encoder are the
single files the user already dropped under ``models/``. Some base modes additionally need a
**training adapter** that undoes turbo distillation while training - it is fused into the base with
the existing LoRA fuser, so the base behaves de-distilled while the trainable LoRA learns on top.

Krea 2's recommended path is different: train on **RAW** (never distilled) and apply the result to
Turbo at generation time. Turbo-plus-adapter exists for people who only hold Turbo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import arch as archs

#: arch -> (env var, category) for each component the trainer resolves itself.
_ENV = {
    archs.Z_IMAGE: {
        "diffusion_models": "INLINE_ZIMAGE_MODEL",
        "vae": "INLINE_ZIMAGE_VAE",
        "text_encoders": "INLINE_ZIMAGE_TEXT_ENCODER",
        "adapter": "INLINE_ZIMAGE_TRAIN_ADAPTER",
    },
    archs.KREA2: {
        "diffusion_models": "INLINE_KREA2_MODEL",
        "vae": "INLINE_KREA2_VAE",
        "text_encoders": "INLINE_KREA2_TEXT_ENCODER",
        "adapter": "INLINE_KREA2_TRAIN_ADAPTER",
    },
    archs.FLUX2: {
        "diffusion_models": "INLINE_FLUX2_MODEL",
        "vae": "INLINE_FLUX2_VAE",
        "text_encoders": "INLINE_FLUX2_TEXT_ENCODER",
        "adapter": "INLINE_FLUX2_TRAIN_ADAPTER",
    },
    archs.MINIMAX_H3: {
        "diffusion_models": "INLINE_MINIMAXH3_MODEL",
        "vae": "INLINE_MINIMAXH3_VIDEO_VAE",
        "text_encoders": "INLINE_MINIMAXH3_TEXT_ENCODER",
        "adapter": "INLINE_MINIMAXH3_TRAIN_ADAPTER",
    },
}


@dataclass
class Encoders:
    """The pieces the one-off latent/caption precache needs. Loaded, used, then freed *before* the
    transformer loads - together they do not fit alongside a 12-26GB transformer."""

    vae: Any
    text_encoder: Any
    tokenizer: Any
    scheduler: Any
    #: Krea 2 only: a transformer-less Krea2Pipeline, so caption encoding goes through diffusers'
    #: own prompt template and 12-layer tap rather than a copy that could drift from inference.
    pipeline: Any = None


def _require(root: Path, arch: str, category: str) -> str:
    """The weight file for a component, resolved by the arch's own requirements module.

    Never "the first file in the folder": every architecture shares ``vae/`` and ``text_encoders/``,
    so alphabetical order silently hands Z-Image the Qwen3-VL encoder once Krea 2 is installed
    alongside it, and the mismatch only surfaces as a meta-tensor error deep inside the load."""
    env = os.environ.get(_ENV[arch][category])
    if env:
        return env
    picked = _resolve(arch, category)
    if picked:
        return str(picked)
    raise RuntimeError(
        f"No {arch} {category} weight found under {root / category}. Add it there "
        f"(or set {_ENV[arch][category]})."
    )


def _resolve(arch: str, category: str) -> Any:
    """The arch's own answer for a component, so training and generation pick the same file."""
    if arch == archs.MINIMAX_H3:
        from ..models.minimaxh3 import requirements as h3_reqs

        if category == "vae":
            return h3_reqs.resolve("vae", h3_reqs.VIDEO_VAE_FILE)
        if category == "text_encoders":
            return h3_reqs.resolve("text_encoders", "MiniMax-H3-text-encoder")
        # Only fl2va trains: ref2va is the same architecture reached through reference conditioning,
        # so a LoRA learned on one loads on the other.
        return h3_reqs.resolve_transformer("fl2va")

    if arch == archs.FLUX2:
        from ..models.flux2 import requirements as flux2_reqs

        if category == "vae":
            return flux2_reqs.resolve_vae(None)
        if category == "text_encoders":
            return flux2_reqs.resolve_text_encoder(None)
        return flux2_reqs.resolve_diffusion(None)

    if arch == archs.KREA2:
        from ..models.krea2 import requirements as krea2_reqs

        if category == "vae":
            return krea2_reqs.resolve_vae(None)
        if category == "text_encoders":
            return krea2_reqs.resolve_text_encoder(None)
        return None

    from ..models.zimage import requirements as zimage_reqs

    if category == "vae":
        return zimage_reqs.resolve_vae(None)
    if category == "text_encoders":
        return zimage_reqs.resolve_text_encoder(None)
    resolved = zimage_reqs.resolve_diffusion(None)
    # Only a single file can be fine-tuned; a whole-pipeline folder has no checkpoint to stream.
    return resolved[1] if resolved and resolved[0] == "single_file" else None



def loader_arch(arch: str, models_dir: str | None = None) -> str:
    """The ``models/loaders.py`` arch key for a training arch.

    They are not the same namespace: training says ``flux2`` while the loaders key their config and
    tokenizer bundles per variant (``flux2-klein-4b``, ``flux2-dev``), because a 4B and a 9B need
    different encoder configs. Resolved from whichever checkpoint is installed.
    """
    if arch != archs.FLUX2:
        return arch
    from ..models.flux2 import requirements as flux2_reqs

    variant = flux2_reqs.resolved_variant(None)
    return variant.arch if variant is not None else "flux2-klein-4b"


def _adapter_path(root: Path, arch: str, base_mode: str) -> str | None:
    """The de-distillation adapter for a turbo base mode, or None when the base is undistilled."""
    if base_mode not in ("turbo_adapter",):
        return None
    if arch == archs.FLUX2:
        raise RuntimeError(
            "FLUX.2 has no de-distillation adapter. Train against a -base- checkpoint instead; "
            "the adapter it produces still loads on the distilled build afterwards."
        )
    picked = os.environ.get(_ENV[arch]["adapter"])
    if not picked:
        loras = root / "loras"
        if loras.is_dir():
            candidates = [
                p for p in sorted(loras.iterdir()) if "adapter" in p.name.lower()
            ]
            if arch == archs.KREA2:
                candidates = [p for p in candidates if "krea" in p.name.lower()]
            else:
                candidates = [p for p in candidates if "krea" not in p.name.lower()]
            picked = str(candidates[0]) if candidates else None
    if not picked:
        # Named explicitly: the adapter is the one component no model popup offers, so an error
        # that only says "add an adapter" leaves the user with nowhere to go.
        raise RuntimeError(
            f"Turbo mode needs a training adapter to avoid turbo drift. Download "
            f"{_ADAPTER_SOURCE[arch]} into models/loras/ (or set {_ENV[arch]['adapter']}), or "
            f"train on the undistilled base instead."
        )
    return picked


#: Where each arch's de-distillation adapter comes from. Not served by the model popup, which only
#: covers the diffusion model, VAE and text encoder.
_ADAPTER_SOURCE = {
    archs.Z_IMAGE: "ostris/zimage_turbo_training_adapter",
    archs.KREA2: "ostris/krea2_turbo_training_adapter",
}


def compute_dtype() -> Any:
    """bf16 wherever torch will take it, deliberately including Turing.

    The device policy prefers fp16 below compute 8.0, where bf16 has no tensor cores. That argument
    is about GPU matmuls and does not transfer here: this dtype also reaches the caption pass, which
    runs on the CPU when the text encoder will not fit the card, and CPU fp16 upcasts. Switching a
    T4 to fp16 hung the machine mid-caption. Narrow it to the GPU compute before revisiting.
    """
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_encoders(models_dir: str, arch: str, device: str, dtype: Any) -> Encoders:
    """The VAE + text encoder + tokenizer + scheduler, for the precache pass."""
    from ..models import loaders

    root = Path(models_dir)
    vae_file = _require(root, arch, "vae")
    encoder_file = _require(root, arch, "text_encoders")

    if arch == archs.KREA2:
        from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline

        vae = loaders.load_qwen_image_vae(vae_file, dtype, device=device)
        text_encoder, tokenizer = loaders.load_qwen3vl_text_encoder(
            encoder_file, dtype, device=device
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(loaders.KREA2_SCHEDULER_CONFIG)
        # Transformer-less: only encode_prompt is used, and building the 26GB base here would
        # defeat the whole point of precaching before it loads.
        pipeline = Krea2Pipeline(
            scheduler=scheduler, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=None,
        )
        return Encoders(vae, text_encoder, tokenizer, scheduler, pipeline)

    if arch == archs.FLUX2:
        from diffusers import Flux2KleinPipeline

        larch = loader_arch(arch, models_dir)
        vae = loaders.load_flux2_vae(larch, vae_file, dtype, device=device)
        text_encoder, tokenizer = loaders.load_text_encoder(
            larch, encoder_file, dtype, device=device
        )
        scheduler = loaders.load_scheduler(larch)
        # Transformer-less, so caption encoding goes through diffusers' own Qwen3 chat template and
        # three-layer tap rather than a copy here that could drift from inference.
        pipeline = Flux2KleinPipeline(
            scheduler=scheduler, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=None,
        )
        return Encoders(vae, text_encoder, tokenizer, scheduler, pipeline)

    vae = loaders.load_vae(arch, vae_file, dtype, device=device)
    text_encoder, tokenizer = loaders.load_text_encoder(arch, encoder_file, dtype, device=device)
    return Encoders(vae, text_encoder, tokenizer, loaders.load_scheduler(arch))


def free_encoders(encoders: Encoders) -> None:
    """Return the VAE + text-encoder VRAM once latents/captions are cached.

    Dropped, not moved to CPU: a RAM-tight host can't take the encoder either. The scheduler is
    config-only, so it stays."""
    import gc

    import torch

    from ..models import loaders

    encoders.vae = None
    encoders.text_encoder = None
    encoders.tokenizer = None
    encoders.pipeline = None
    loaders.unload_components(keep_files=set())  # the transformer isn't loaded yet - drop it all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: Architectures whose loader can build a 4-bit base. Z-Image is not here because it does not need
#: to be: it trains in ~15GB at 1024, so bf16 already fits the cards people have.
_QUANTIZABLE = {archs.KREA2, archs.FLUX2}

#: Peak activation cost per image token, measured at rank 16, batch 1, gradient checkpointing on.
#: Krea 2's wider blocks cost roughly 7x Z-Image's per token, and activations - not weights - are
#: what decides whether 1024 fits.
_ACTIVATION_MB_PER_TOKEN = {archs.KREA2: 5.2, archs.Z_IMAGE: 0.8, archs.FLUX2: 1.6}

#: Room left for the adapter, its 8-bit optimizer state and allocator slack.
_MARGIN_BYTES = 2 * 1024**3


def resolve_quant(
    base_quant: str, models_dir: str, arch: str, base_mode: str, resolution: int
) -> Any:
    """The base-weight quantization to train under.

    ``auto`` is the default and the point of the setting: Krea 2's base is 26GB at bf16, so a card
    that cannot hold it *plus its activations* trains in NF4 instead of failing. The adapter stays
    full precision either way (the QLoRA arrangement), so only the frozen base loses precision.

    Resolution is part of the decision because it dominates it. A 46GB card holds the 26GB base
    fine, then OOMs at 1024 where the activations alone want ~21GB."""
    from ..device.policy import Quantization

    if arch == archs.MINIMAX_H3:
        # 4-bit is not a rung on a ladder for H3, it is the only way it loads: 40GB of base after
        # the AdaLN factorisation, before activations or the adapter, does not leave room on any
        # card this trains on. Refused rather than offered and then failing hours in.
        if base_quant == "none":
            raise RuntimeError(
                "MiniMax H3 has no full-precision training path. Its base is 40GB after the AdaLN "
                "factorisation, so it trains in 4-bit. Set the base precision back to auto."
            )
        return Quantization.NF4

    if base_quant == "none":
        return Quantization.NONE
    if base_quant == "nf4":
        if arch not in _QUANTIZABLE:
            raise RuntimeError(
                f"{arch} has no 4-bit training path; it trains in bf16. Set the base precision "
                "back to auto."
            )
        return Quantization.NF4
    if base_quant != "auto":
        raise RuntimeError(f"Unknown base quantization {base_quant!r}.")

    import torch

    if not torch.cuda.is_available() or arch not in _QUANTIZABLE:
        return Quantization.NONE
    base = _base_size(models_dir, arch, base_mode)
    if not base:
        return Quantization.NONE  # unmeasurable, so do not guess at the user's expense
    needed = base + _activation_bytes(arch, resolution) + _MARGIN_BYTES
    fits = torch.cuda.get_device_properties(0).total_memory >= needed
    return Quantization.NONE if fits else Quantization.NF4


def resolve_offload(
    pref: str, quant: Any, models_dir: str, arch: str, base_mode: str, resolution: int
) -> bool:
    """Whether to stream saved activations to host RAM this run.

    Only meaningful for a full-precision (bf16) base: the point is to keep the 26GB Krea 2 base
    resident and fit the ~21GB of 1024 activations elsewhere, rather than dropping the frozen base
    to NF4. A quantized base already fits, so offload just adds PCIe traffic for nothing. ``auto``
    turns it on exactly when the bf16 plan would otherwise not fit VRAM; ``on``/``off`` force it."""
    from ..device.policy import Quantization

    if pref == "off":
        return False
    if quant is not Quantization.NONE:
        return False  # a quantized base already fits; offload would only slow it down
    if pref == "on":
        return True
    if pref not in ("auto", ""):
        raise RuntimeError(f"Unknown offload preference {pref!r}.")

    import torch

    if not torch.cuda.is_available():
        return False
    base = _base_size(models_dir, arch, base_mode)
    if not base:
        return False  # unmeasurable, so do not pay the offload cost on a guess
    needed = base + _activation_bytes(arch, resolution) + _MARGIN_BYTES
    return torch.cuda.get_device_properties(0).total_memory < needed


def _activation_bytes(arch: str, resolution: int) -> int:
    """Estimated peak activation memory. Image tokens are the VAE's 8x downscale then 2x2
    patching, and cost is linear in them - attention is memory-efficient, so there is no square."""
    tokens = (max(resolution, 1) // 16) ** 2
    per_token = _ACTIVATION_MB_PER_TOKEN.get(arch, 5.2)
    return int(tokens * per_token * 1024**2)


def _base_size(models_dir: str, arch: str, base_mode: str) -> int:
    """The base checkpoint's size on disk, which is its resident size at bf16. 0 if unmeasurable."""
    try:
        return Path(_base_file(Path(models_dir), arch, base_mode)).stat().st_size
    except (OSError, RuntimeError):
        return 0


def _proc_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _memory_totals() -> tuple[int, int]:
    """``(RAM, swap)`` in bytes, or ``(0, 0)`` where /proc/meminfo is not readable."""
    found: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "SwapTotal"):
                found[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return 0, 0
    return found.get("MemTotal", 0), found.get("SwapTotal", 0)


def check_base_mappable(models_dir: str, arch: str, base_mode: str) -> None:
    """Refuse a run the kernel will not let mmap the base, before the precache rather than after.

    safetensors maps the checkpoint in one call, so a 62GB file asks for a 62GB mapping. Under
    ``vm.overcommit_memory=0`` the kernel rejects any single mapping larger than RAM plus swap, and
    the error it raises names the checkpoint, so it reads like a corrupt download. The mapping is
    virtual and the loader streams through it at roughly one tensor of resident memory, so the
    limit is bookkeeping rather than a real shortage. Checked here because precaching a large
    dataset costs twenty minutes and runs first: the cheap failure has to come before the dear one.
    """
    mode = _proc_int("/proc/sys/vm/overcommit_memory")
    # 1 is unrestricted, and a missing knob means this is not Linux.
    if mode is None or mode == 1:
        return
    size = _base_size(models_dir, arch, base_mode)
    ram, swap = _memory_totals()
    if size <= 0 or ram <= 0:
        return
    if mode == 2:
        ratio = _proc_int("/proc/sys/vm/overcommit_ratio") or 50
        allowed = ram * ratio // 100 + swap
    else:
        allowed = ram + swap
    if size <= allowed:
        return

    gib = 1024**3
    name = Path(_base_file(Path(models_dir), arch, base_mode)).name
    raise RuntimeError(
        f"{name} needs a single {size / gib:.1f}GiB memory mapping, but this machine caps one at "
        f"{allowed / gib:.1f}GiB (RAM {ram / gib:.0f}GiB plus swap {swap / gib:.0f}GiB) while "
        f"vm.overcommit_memory={mode}. The mapping is virtual and the weights stream through it, "
        f"so the memory is never all used at once, but the kernel refuses the request up front. "
        f"Allow it with:\n"
        f"    sudo sysctl -w vm.overcommit_memory=1\n"
        f"and to keep it across reboots:\n"
        f"    echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-inline-studio.conf"
    )


def load_transformer(
    models_dir: str, arch: str, base_mode: str, device: str, dtype: Any, quant: Any = None
) -> Any:
    """The base transformer (frozen, optionally quantized), with a training adapter fused first."""
    from ..device.policy import Quantization
    from ..graph.loader_runners import LoraRef
    from ..models import loaders

    quant = quant or Quantization.NONE

    if arch == archs.MINIMAX_H3:
        from . import h3

        # No de-distillation adapter and no base-mode branch: H3 ships one undistilled build per
        # partition, and only fl2va trains.
        del base_mode
        return h3.load_base(models_dir, device, dtype, quant)

    root = Path(models_dir)
    adapter = _adapter_path(root, arch, base_mode)
    loras: tuple[LoraRef, ...] = (LoraRef(file=adapter, strength=1.0),) if adapter else ()

    diffusion = _base_file(root, arch, base_mode)
    if arch == archs.FLUX2:
        from ..models.checkpoint import CheckpointReader
        from ..models.flux2 import variants as flux2_variants

        config = flux2_variants.derive_transformer_config(CheckpointReader(diffusion).shapes())
        if config is None:
            raise RuntimeError(f"{Path(diffusion).name} is not a FLUX.2 checkpoint.")
        return loaders.load_flux2_transformer(
            loader_arch(arch, models_dir), diffusion, config, dtype, quant,
            device=device, loras=loras,
        )
    if arch == archs.KREA2:
        return loaders.load_krea2_transformer(
            diffusion, dtype, quant, device=device, loras=loras
        )
    return loaders.load_diffusion(arch, diffusion, dtype, quant, device=device, loras=loras)


def _base_file(root: Path, arch: str, base_mode: str) -> str:
    """The base checkpoint this run trains against."""
    if arch == archs.FLUX2:
        return _flux2_base_file(root)
    if arch != archs.KREA2:
        return _require(root, arch, "diffusion_models")

    from ..models.krea2 import requirements as reqs

    variant = "turbo" if base_mode == "turbo_adapter" else "raw"
    override = os.environ.get(_ENV[arch]["diffusion_models"])
    diffusion = override or reqs.resolve_diffusion(variant)
    if not diffusion:
        raise RuntimeError(
            f"No Krea 2 {variant.upper()} checkpoint found under {root / 'diffusion_models'}. "
            f"Add {reqs.DIFFUSION_FILES[variant]} there."
        )
    return str(diffusion)


def _flux2_base_file(root: Path) -> str:
    """The **undistilled** FLUX.2 checkpoint to train against.

    Training on a step-distilled build is the documented cause of the collapse reports: BFL and
    musubi-tuner both say to train on ``-base-`` and load the adapter onto the distilled model
    afterwards, which is also faster and usually better. So a distilled checkpoint is refused here
    rather than silently producing a bad LoRA hours later.
    """
    from ..models.flux2 import variants as flux2_variants

    override = os.environ.get(_ENV[archs.FLUX2]["diffusion_models"])
    if override:
        return override
    folder = root / "diffusion_models"
    candidates = sorted(folder.iterdir()) if folder.is_dir() else []
    detected = [(p, flux2_variants.detect(p)) for p in candidates if p.is_file()]
    base = [p for p, v in detected if v is not None and not v.distilled]
    if base:
        return str(base[0])
    distilled = [(p, v) for p, v in detected if v is not None]
    if distilled:
        name, variant = distilled[0][0].name, distilled[0][1]
        raise RuntimeError(
            f"{name} is the step-distilled FLUX.2 {variant.label} build, which trains badly. "
            "Download the matching Base checkpoint from the FLUX.2 node's model popup and train "
            "on that - the LoRA still loads on the distilled build for generation."
        )
    raise RuntimeError(
        f"No FLUX.2 checkpoint found under {folder}. Download one from the node's model popup "
        f"(or set {_ENV[archs.FLUX2]['diffusion_models']})."
    )
