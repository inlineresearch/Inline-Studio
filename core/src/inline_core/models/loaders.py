"""Model-agnostic single-file loaders - assemble a diffusers pipeline from individual weight files
(ComfyUI-style), never touching the network at generation time.

The problem this solves: loading a whole diffusers *pipeline folder* with ``from_pretrained`` is
slow (sharded weights reconstructed from an index, per-component metadata resolution, and a stall if
any config is not local). ComfyUI is fast because it loads **one consolidated ``.safetensors`` per
component**, builds each model from a config it ships, and bundles the tokenizer - zero hub
round-trip. This module does the same:

  - the big weights come from the user's three files (``diffusion_models/`` / ``vae/`` /
    ``text_encoders/``), loaded via ``from_single_file`` / a config-driven ``state_dict`` load;
  - the small configs + the Qwen tokenizer come from a **fetch-once** asset bundle
    (``ensure_assets``), pulled once into the engine data dir and then reused offline.

An ``ArchSpec`` maps a model family to its reference repo (for the small assets) and to how each
component is built. Only ``z-image`` is wired today; Flux and friends slot in as new specs - the
Z-Image node's dropdowns and the decomposed ``load/*`` subnodes both call through here, so a new
arch is a data change, not new plumbing.

Heavy deps (torch, diffusers, transformers, huggingface_hub, safetensors) are imported **inside**
functions on purpose: importing this module stays cheap, and a torch-less engine boot never trips
over it. Callers are the model-runner subpackages, which the server registers best-effort.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import data_dir
from ..device.policy import Quantization
from ..errors import ComponentError
from .lora import fuse_loras

if TYPE_CHECKING:
    from ..graph.loader_runners import LoraRef


@dataclass(frozen=True)
class ArchSpec:
    """How to source the small offline assets (configs + tokenizer) for one model family.

    ``asset_files`` are repo-relative paths under ``assets_repo`` - the tiny config/tokenizer files
    (never the multi-GB weights) fetched once into ``data_dir()/assets/<key>``. The subfolder layout
    is preserved, so a component loads its config with ``config=<assets_root>, subfolder="<name>"``.
    """

    key: str
    assets_repo: str
    asset_files: tuple[str, ...]


#: The reference repo carries the small diffusers configs + the Qwen3 tokenizer. Weights are never
#: taken from here - the user supplies those as single files (see requirements.py).
_ZIMAGE = ArchSpec(
    key="z-image",
    assets_repo="Tongyi-MAI/Z-Image-Turbo",
    asset_files=(
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
    ),
)

SPECS: dict[str, ArchSpec] = {_ZIMAGE.key: _ZIMAGE}


def _spec(arch: str) -> ArchSpec:
    spec = SPECS.get(arch)
    if spec is None:
        raise ComponentError(f"No loader registered for architecture {arch!r}.")
    return spec


# --- fetch-once assets --------------------------------------------------------------------------

_ASSETS_LOCK = Lock()


def assets_root(arch: str) -> Path:
    """Where an arch's small configs/tokenizer live once fetched (engine-owned, offline after)."""
    return data_dir() / "assets" / arch


def ensure_assets(arch: str) -> Path:
    """Make the arch's config + tokenizer files present locally, fetching them **once** (~15 MB of
    small files, never the weights) from the reference repo. Idempotent: a ``.complete`` marker
    short-circuits every later call, so generation stays fully offline after the first run.

    Raises ComponentError if the first fetch can't reach the hub - the assets are the only piece not
    supplied by the user's single files, so we surface it clearly instead of a deep hub traceback.
    """
    spec = _spec(arch)
    root = assets_root(arch)
    marker = root / ".complete"
    if marker.is_file():
        return root
    with _ASSETS_LOCK:
        if marker.is_file():
            return root
        try:
            from huggingface_hub import hf_hub_download

            for rel in spec.asset_files:
                hf_hub_download(spec.assets_repo, rel, local_dir=str(root))
        except Exception as error:  # noqa: BLE001 - re-raised as a clear component error
            raise ComponentError(
                f"Could not fetch the one-time {arch} config/tokenizer assets from "
                f"{spec.assets_repo} ({error}). This ~15 MB download happens once; after it, "
                "generation runs fully offline."
            ) from error
        marker.write_text("ok")
    return root


def _link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` resolve to ``src``'s bytes, preferring a symlink, then a hardlink, then a copy -
    so the staging dir costs ~zero on Linux but still works where symlinks/hardlinks are unavailable
    (Windows without privilege, cross-device)."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _staged_encoder_dir(arch: str, file: str) -> Path:
    """A tiny engine-owned dir transformers can load the text encoder from as a normal model: the
    bundled config next to the user's weights file linked in as ``model.safetensors``.

    Why: passing a pre-materialized ``state_dict`` to ``from_pretrained`` forces the whole ~8 GB
    encoder into CPU RAM and BYPASSES transformers' native ``safe_open(..., backend="mmap")`` →
    device streaming loader. Loading from a directory instead lets that loader run, so peak host RAM
    stays ≈ one tensor. Idempotent via a ``.complete`` marker; keyed by the weights path so a
    different file stages afresh."""
    root = ensure_assets(arch)
    te_config = root / "text_encoder"
    digest = hashlib.sha1(str(file).encode()).hexdigest()[:16]
    stage = assets_root(arch) / "te_stage" / digest
    marker = stage / ".complete"
    if marker.is_file():
        return stage
    with _ASSETS_LOCK:
        if marker.is_file():
            return stage
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "generation_config.json"):
            src = te_config / name
            if src.is_file():
                _link_or_copy(src, stage / name)
        _link_or_copy(Path(file), stage / "model.safetensors")
        marker.write_text("ok")
    return stage


def _release_transient() -> None:
    """Drop the just-loaded checkpoint's transient buffers (mmap views, freed CUDA blocks) so the
    next big component starts from a clean allocator - keeps peak VRAM/RAM near one component, not
    the sum of all three. Best-effort; never breaks a load."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache hygiene must never break a load
        pass


# --- quantization ------------------------------------------------------------------------------


def _quant_config(quant: Quantization, *, framework: str) -> Any | None:
    """A diffusers/transformers ``quantization_config`` for the requested weight quantization, or
    None for full precision. ``framework`` picks which library's config class to build:
    ``"diffusers"`` for the transformer, ``"transformers"`` for the Qwen3 text encoder.

    INT8 uses torch-native weight-only quantization (torchao) on purpose: unlike bitsandbytes it
    stays a movable tensor subclass, so it coexists with ``enable_model_cpu_offload`` - exactly the
    smart-memory case that spreads the model across VRAM + RAM. NF4 (bitsandbytes) is CUDA-only.
    Best-effort: if the optional backend (torchao/bitsandbytes) is not installed we return None and
    load full precision rather than crash - the caller (smart memory) still gets CPU offload."""
    if quant is Quantization.NONE:
        return None
    try:
        if quant is Quantization.INT8:
            # torchao dropped the string form ("int8_weight_only") in 0.14+; pass the config object.
            from torchao.quantization import Int8WeightOnlyConfig

            aoc = Int8WeightOnlyConfig()
            if framework == "diffusers":
                from diffusers import TorchAoConfig as DiffusersTorchAoConfig

                return DiffusersTorchAoConfig(aoc)
            from transformers import TorchAoConfig as TransformersTorchAoConfig

            return TransformersTorchAoConfig(aoc)
        # NF4
        if framework == "diffusers":
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

            return DiffusersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        return TransformersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    except Exception:  # noqa: BLE001 - a missing/incompatible quant backend must not break loading
        import logging

        logging.getLogger("inline_core.zimage").warning(
            "Quantization %s (%s) unavailable - loading full precision. Install torchao (int8) or "
            "bitsandbytes (nf4) to shrink weights for smart memory.",
            quant.value,
            framework,
        )
        return None


def _quantize_in_place(model: Any, quant: Quantization) -> None:
    """Apply torchao weight-only quantization to an already-built model, in place. Used for the
    diffusion transformer, whose ``from_single_file`` loader silently ignores a ``quantization_config``
    (so the config-based path leaves it full-size). torchao's ``quantize_`` swaps each ``nn.Linear``
    weight for an int8 tensor subclass on whatever device the module already sits on.

    Best-effort: a missing/incompatible torchao is logged and left full precision rather than crashing
    the load - the fit estimate that chose int8 will then be optimistic, but that surfaces as a normal
    OOM node error, not a hard import failure. Only INT8 is handled here; NF4 stays config-driven."""
    if quant is not Quantization.INT8:
        return
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        quantize_(model, Int8WeightOnlyConfig())
    except Exception:  # noqa: BLE001 - a missing/incompatible quant backend must not break loading
        import logging

        logging.getLogger("inline_core.zimage").warning(
            "torchao int8 quantization of the transformer failed - loading full precision. Install "
            "a compatible torchao to shrink the weights for a tight GPU.",
        )


# --- component loaders (cached in-process) ------------------------------------------------------

# Keyed by (arch, kind, file, dtype, quant, device[, loras]) so switching one file (e.g. a different
# VAE), the quantization, or the load device reuses the other already-loaded components. The run
# manager executes one run at a time; the lock guards each build.
#
# The LoRA stack MUST be part of the key: a fused model is a different artifact from the same file
# unfused, and serving one for the other is silent - the output is plausible but wrong.
_CACHE: dict[tuple[str, ...], Any] = {}
_CACHE_LOCK = Lock()


def _cached(key: tuple[str, ...], build: Callable[[], Any]) -> Any:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        value = build()
        _CACHE[key] = value
        return value


def _dtype_key(dtype: Any) -> str:
    return str(dtype)


def _device_key(device: str | None) -> str:
    return device or "cpu"


def lora_cache_key(loras: tuple[LoraRef, ...]) -> tuple[str, ...]:
    """Empty stack -> no extra key elements, so an un-LoRA'd load keeps its original cache key."""
    return tuple(f"{lora.file}@{lora.strength:g}" for lora in loras)


def unload_components(
    keep_files: set[str] | None = None, keep_loras: tuple[str, ...] | None = None
) -> None:
    """Drop cached components whose source file is NOT in ``keep_files``, freeing their VRAM/RAM.

    Called when switching checkpoints so a new model doesn't stack on top of the previous one (the
    cache never evicted before, so a second distinct model roughly doubled resident memory). Only
    drops references + empties the CUDA cache - it does not move weights to CPU RAM (that would just
    relocate the pressure on a RAM-tight box). The caller must drop any pipeline holding these
    components first, or the references keep them alive.

    ``keep_loras`` additionally evicts *other LoRA variants of a kept file*. Without it, fusing a
    LoRA into an already-loaded checkpoint keeps the unfused transformer (same file) and builds a
    second one alongside it - two full-size models, which OOMs a 16 GB card."""
    keep = keep_files or set()
    with _CACHE_LOCK:
        # k = (arch, kind, file, dtype, quant, device, *loras)
        stale = [
            k
            for k in _CACHE
            if k[2] not in keep
            or (keep_loras is not None and k[1] == "diffusion" and k[6:] != keep_loras)
        ]
        for k in stale:
            comp = _CACHE.pop(k)
            del comp
    _release_transient()


def load_diffusion(
    arch: str,
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """The diffusion transformer from a single ``.safetensors``. diffusers converts the checkpoint
    keys; the config comes from the bundled assets, so nothing is fetched at load time. ``quant``
    (smart memory) quantizes the weights on load. ``device`` (e.g. ``"cuda:0"``) streams each tensor
    **straight to the GPU** from an mmap-backed checkpoint - the fp16 weights are never materialized
    as an anonymous CPU copy (the host-RAM spike that OOM-killed the server), and for the int8 path
    torchao quantizes on-device per tensor. ``None`` loads to CPU (the offload path, where
    accelerate installs its hooks before placing)."""

    def build() -> Any:
        from diffusers import ZImageTransformer2DModel

        root = ensure_assets(arch)
        model = ZImageTransformer2DModel.from_single_file(
            file,
            config=str(root),
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,  # meta-init; needed for device= to stream (already the default)
            device=device,
            local_files_only=True,
        )
        # diffusers' ``from_single_file`` **ignores** ``quantization_config`` (unlike
        # ``from_pretrained``), so the transformer would load at full size and the "int8" plan would
        # blow the VRAM budget (a T4 OOMs mid-load). Quantize it explicitly with torchao after the
        # load instead - the weights briefly sit full-size on the device, then halve in place.
        # LoRAs fuse *before* quantization: merging into already-int8 weights would quantize twice.
        fuse_loras(model, loras)
        _quantize_in_place(model, quant)
        return model

    return _cached(
        (arch, "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device))
        + lora_cache_key(loras),
        build,
    )


def load_vae(arch: str, file: str, dtype: Any, device: str | None = None) -> Any:
    """The VAE from a single ``.safetensors`` (the Flux/LDM-style ``ae.safetensors``). diffusers'
    LDM-VAE converter remaps the keys; the config is the bundled ``AutoencoderKL`` config.
    ``device`` streams it straight to the GPU (small, but keeps every component off the CPU on the
    resident path)."""

    def build() -> Any:
        from diffusers import AutoencoderKL

        root = ensure_assets(arch)
        return AutoencoderKL.from_single_file(
            file,
            config=str(root),
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device=device,
            local_files_only=True,
        )

    # The VAE stays full precision (it is small - a few hundred MB - and int8 on the conv VAE costs
    # quality for no meaningful memory win); the key still carries a quant slot for a uniform shape.
    return _cached(
        (arch, "vae", file, _dtype_key(dtype), Quantization.NONE.value, _device_key(device)), build
    )


def load_text_encoder(
    arch: str,
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """The Qwen3 text encoder + tokenizer. The weights come from the user's single file; the config
    and tokenizer come from the bundled assets.

    We load from a tiny staging directory (config next to the weights, see ``_staged_encoder_dir``)
    rather than passing a pre-loaded ``state_dict``: passing a state_dict forces the entire ~8 GB
    encoder into a CPU RAM dict and bypasses transformers' native mmap → device streaming loader.
    From a directory + ``device_map={"": device}``, transformers materializes each tensor lazily
    onto the GPU, so peak host RAM stays ≈ one tensor. ``quant`` (smart memory) int8-quantizes the
    encoder - the largest single weight - on load. ``device=None`` loads to CPU for the accelerate
    offload path."""

    def build() -> tuple[Any, Any]:
        from transformers import AutoTokenizer, Qwen3Model

        root = ensure_assets(arch)
        weights_dir = _staged_encoder_dir(arch, file)
        device_map = {"": device} if device else None
        text_encoder = Qwen3Model.from_pretrained(
            str(weights_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            local_files_only=True,
            quantization_config=_quant_config(quant, framework="transformers"),
        )
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        return text_encoder, tokenizer

    return _cached(
        (arch, "text_encoder", file, _dtype_key(dtype), quant.value, _device_key(device)), build
    )


def load_scheduler(arch: str) -> Any:
    """The flow-match scheduler, from the bundled config (config-only - never downloads)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    root = ensure_assets(arch)
    try:
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(root), subfolder="scheduler", local_files_only=True
        )
    except (OSError, ValueError):
        return FlowMatchEulerDiscreteScheduler()


def assemble_zimage_pipeline(
    *,
    diffusion_file: str,
    vae_file: str,
    text_encoder_file: str,
    dtype: Any,
    img2img: bool,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    cancel_check: Callable[[], None] | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """Build a Z-Image pipeline from three local single files. Components are cached individually,
    so swapping one file reuses the others. The returned pipeline is unplaced - the runner owns
    device placement / low-VRAM tweaks. ``quant`` (smart memory) quantizes the big weights (the
    transformer + text encoder) on load. ``vae_dtype`` (defaults to ``dtype``) lets the VAE keep a
    safer dtype than the denoiser - e.g. fp32 when the transformer runs fp16 (whose decode can
    overflow). ``device`` (e.g. ``"cuda:0"``) streams each component straight to the GPU so peak
    host RAM never holds a full component; ``None`` loads to CPU for the offload path. Buffers are
    released between the three big loads so peak memory tracks one component, not their sum.
    ``cancel_check`` (if given) is called between the three loads so an interrupt during the multi-
    second load bails after the current component instead of only at the first denoise step. It
    cannot break into a single component's blocking read (the transformer dominates), but it stops
    the run before loading the VAE + text encoder + placing + denoising."""
    from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

    arch = _ZIMAGE.key
    transformer = load_diffusion(arch, diffusion_file, dtype, quant, device=device, loras=loras)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    vae = load_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    text_encoder, tokenizer = load_text_encoder(
        arch, text_encoder_file, dtype, quant, device=device
    )
    _release_transient()
    scheduler = load_scheduler(arch)
    cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
    return cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
