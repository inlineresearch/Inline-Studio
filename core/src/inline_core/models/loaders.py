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

if TYPE_CHECKING:
    from ..graph.loader_runners import LoraRef


@dataclass(frozen=True)
class AssetFile:
    """One small offline asset. ``repo`` overrides the spec's default (Krea 2 takes its tokenizer
    from Qwen3-VL and its VAE config from Qwen-Image); ``local`` overrides where it lands when the
    source repo's layout differs from the subfolder a component loads its config from."""

    path: str
    repo: str | None = None
    local: str | None = None

    def target(self, spec: ArchSpec) -> tuple[str, str]:
        return (self.repo or spec.assets_repo, self.local or self.path)


@dataclass(frozen=True)
class ArchSpec:
    """How to source the small offline assets (configs + tokenizer) for one model family.

    The tiny config/tokenizer files (never the multi-GB weights) are fetched once into
    ``data_dir()/assets/<key>``. The subfolder layout is preserved, so a component loads its config
    with ``config=<assets_root>, subfolder="<name>"``.
    """

    key: str
    assets_repo: str
    asset_files: tuple[AssetFile, ...]


#: The reference repo carries the small diffusers configs + the Qwen3 tokenizer. Weights are never
#: taken from here - the user supplies those as single files (see requirements.py).
_ZIMAGE = ArchSpec(
    key="z-image",
    assets_repo="Tongyi-MAI/Z-Image-Turbo",
    asset_files=tuple(
        AssetFile(path)
        for path in (
            "transformer/config.json",
            "vae/config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/merges.txt",
            "tokenizer/vocab.json",
        )
    ),
)

#: Krea 2's own repos are gated, so the assets come from the two ungated Qwen repos its components
#: are built from. The transformer + scheduler configs are code constants (see `krea2/config.py`),
#: so nothing here needs the Krea 2 repo at all.
_KREA2 = ArchSpec(
    key="krea2",
    assets_repo="Qwen/Qwen3-VL-4B-Instruct",
    asset_files=(
        AssetFile("config.json", local="text_encoder/config.json"),
        AssetFile("tokenizer.json", local="tokenizer/tokenizer.json"),
        AssetFile("tokenizer_config.json", local="tokenizer/tokenizer_config.json"),
        AssetFile("vocab.json", local="tokenizer/vocab.json"),
        AssetFile("merges.txt", local="tokenizer/merges.txt"),
        AssetFile("vae/config.json", repo="Qwen/Qwen-Image"),
    ),
)

SPECS: dict[str, ArchSpec] = {_ZIMAGE.key: _ZIMAGE, _KREA2.key: _KREA2}


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
            import shutil

            from huggingface_hub import hf_hub_download

            for asset in spec.asset_files:
                repo, local = asset.target(spec)
                # token=False: these repos are public, and a stale ambient token 401s even on those.
                fetched = hf_hub_download(repo, asset.path, local_dir=str(root), token=False)
                if local != asset.path:
                    destination = root / local
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(fetched, destination)
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
    diffusion transformer, whose ``from_single_file`` loader silently ignores a
    ``quantization_config`` (so the config-based path leaves it full-size). torchao's ``quantize_``
    swaps each ``nn.Linear`` weight for an int8 tensor subclass on the device it already sits on.

    Best-effort: a missing/incompatible torchao is logged and left full precision rather than
    crashing the load - the fit estimate that chose int8 is then optimistic, but that surfaces as a
    normal OOM node error, not a hard import failure. Only INT8 here; NF4 stays config-driven."""
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

# Keyed by (arch, kind, file, dtype, quant, device) so switching one file (e.g. a different VAE),
# the quantization, or the load device reuses the other already-loaded components. A fused diffusion
# transformer appends its LoRA stack (see ``lora_cache_key``) to that base key, so the same file
# with a different stack is a distinct entry. The run manager executes one run at a time; the lock
# guards each build.
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
    """The cache-key suffix a LoRA stack contributes to a fused diffusion transformer. Order- and
    strength-sensitive (fusing is not commutative), keyed by file+strength; an empty stack adds
    nothing, so an un-LoRA'd load keys exactly as it did before LoRA existed."""
    return tuple(f"{lora.file}@{lora.strength}" for lora in loras)


#: Component kinds whose cache key carries a real quantization (the others always store NONE, so
#: comparing their quant slot against the plan's would evict them every time.)
_QUANTIZED_KINDS = ("diffusion", "text_encoder")


def unload_components(
    keep_files: set[str] | None = None,
    keep_loras: tuple[str, ...] | None = None,
    keep_quant: str | None = None,
) -> None:
    """Drop cached components whose source file is NOT in ``keep_files``, freeing their VRAM/RAM.

    Called when switching checkpoints so a new model doesn't stack on top of the previous one (the
    cache never evicted before, so a second distinct model roughly doubled resident memory). Only
    drops references + empties the CUDA cache - it does not move weights to CPU RAM (that would just
    relocate the pressure on a RAM-tight box). The caller must drop any pipeline holding these
    components first, or the references keep them alive.

    ``keep_loras`` (the LoRA-stack suffix being loaded, from ``lora_cache_key``) additionally evicts
    a kept file's diffusion transformer when it carries a *different* stack: fusing a LoRA into an
    already-resident checkpoint would otherwise keep the unfused transformer AND the fused one - two
    full-size models on the card. Left None, eviction is file-only (the pre-LoRA behaviour).

    ``keep_quant`` does the same across quantizations: the fit plan re-quantizes when the footprint
    changes (adding a ControlNet flips a resident load to int8), and the same file at two quants is
    two cache entries, so without this the int8 and full-size transformers both stay resident."""
    keep = keep_files or set()
    with _CACHE_LOCK:
        stale = []
        for k in _CACHE:  # k = (arch, kind, file, dtype, quant, dev, *lora_suffix)
            if k[2] not in keep:
                stale.append(k)
            elif keep_loras is not None and k[1] == "diffusion" and tuple(k[6:]) != keep_loras:
                stale.append(k)
            elif keep_quant is not None and k[1] in _QUANTIZED_KINDS and k[4] != keep_quant:
                stale.append(k)
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
    accelerate installs its hooks before placing). ``loras`` are fused into the weights in order
    (the stack is part of the cache key), so a fused transformer is itself the cached artifact."""

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
        # Fuse LoRAs BEFORE quantizing: the fuse adds a full-precision delta into each weight, which
        # int8 can't accept in place, and int8-quantized weights aren't a plain tensor to add into.
        if loras:
            from .lora import fuse_loras

            fuse_loras(model, loras)
        _quantize_in_place(model, quant)
        return model

    key = (
        arch, "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device),
        *lora_cache_key(loras),
    )
    return _cached(key, build)


#: The Fun ControlNet Union geometry (15 control layers on even base layers, 2 refiners, 33-channel
#: control latent) - shared by the 2.0/2.1/2601/2602 full-size files. Bundled because a dropped-in
#: single file has no config beside it and ``from_single_file`` would otherwise reach for the
#: reference repo's ``config.json`` - a fetch the engine forbids (``local_files_only``), so an
#: offline load raised "does not appear to have a file named config.json". The **lite** variants
#: apply control to 3 layers instead and need their own config; `_controlnet_layer_count` rejects
#: them with a clear message rather than a raw tensor-shape error.
ZIMAGE_CONTROLNET_CONFIG = {
    "add_control_noise_refiner": "control_noise_refiner",
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "control_in_dim": 33,
    "control_layers_places": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    "control_refiner_layers_places": [0, 1],
    "dim": 3840,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
}


def _controlnet_layer_count(file: str) -> int:
    """How many control layers a ControlNet checkpoint carries, read from its header alone (no
    weights). 0 when the file can't be inspected - the strict load then reports the mismatch."""
    try:
        from safetensors import safe_open

        with safe_open(file, "pt") as handle:
            prefix = "control_layers."
            return len(
                {k.split(".")[1] for k in handle.keys() if k.startswith(prefix)}  # noqa: SIM118
            )
    except Exception:  # noqa: BLE001 - a probe must never be the thing that fails a load
        return 0


def load_zimage_controlnet(file: str, dtype: Any, device: str | None = None) -> Any:
    """The Z-Image ControlNet transformer from a single ``.safetensors`` (e.g. the Fun Union model).

    Built from the bundled config and filled by ``assign=``, so nothing is ever fetched and the
    weights land straight on ``device`` (never a full-size CPU copy first). The pipeline grafts the
    shared transformer layers via ``from_transformer`` at build time. Small next to the base
    transformer, so it is never quantized."""

    def build() -> Any:
        from accelerate import init_empty_weights
        from diffusers import ZImageControlNetModel
        from safetensors.torch import load_file

        expected = len(ZIMAGE_CONTROLNET_CONFIG["control_layers_places"])
        found = _controlnet_layer_count(file)
        if found and found != expected:
            raise ComponentError(
                f"This ControlNet applies control to {found} layers, but Inline Core only bundles "
                f"the {expected}-layer Z-Image Fun ControlNet geometry (the 'lite' variants use "
                "fewer). Use a full-size Fun Union/Tile file."
            )
        with init_empty_weights():
            model = ZImageControlNetModel.from_config(ZIMAGE_CONTROLNET_CONFIG)
        state = load_file(file, device=device or "cpu")
        state = {key: value.to(dtype) for key, value in state.items()}
        # assign= replaces the meta parameters outright; a plain copy_ into meta storage errors.
        model.load_state_dict(state, strict=True, assign=True)
        state.clear()
        _release_transient()
        return model.eval()

    # Never quantized, but the key still carries a quant slot so every kind shares one layout (the
    # eviction sweep indexes into it positionally).
    return _cached(
        ("z-image", "controlnet", file, _dtype_key(dtype), Quantization.NONE.value,
         _device_key(device)),
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


# --- Krea 2 ------------------------------------------------------------------------------------

#: The reference "single_mmdit_large_wide" geometry, shared by RAW and Turbo. These are also
#: ``Krea2Transformer2DModel``'s defaults, kept explicit so a diffusers default change is caught.
KREA2_TRANSFORMER_CONFIG = {
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
}

#: Resolution-aware exponential time shift; the Turbo checkpoint pins mu=1.15 in the pipeline.
KREA2_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}


def load_krea2_transformer(
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
) -> Any:
    """The Krea 2 transformer from a single ComfyUI ``.safetensors``.

    ``Krea2Transformer2DModel`` has no ``from_single_file``, so the checkpoint is renamed into
    diffusers naming (``krea2/convert.py``) and streamed in, **one block at a time**: each block is
    materialized, filled, LoRA-fused and quantized before the next one starts. That bound is what
    lets a 26GB checkpoint load as a 7GB NF4 model on a 16GB card - quantizing the whole model
    afterwards would need the full-precision weights resident first.

    Tensors are read by byte range (``checkpoint.py``) rather than through safetensors' whole-file
    mmap, which Linux refuses outright when the file is larger than RAM."""

    def build() -> Any:
        import torch
        from diffusers import Krea2Transformer2DModel

        from . import lora as lora_module
        from .checkpoint import CheckpointReader
        from .krea2 import convert

        with torch.device("meta"):
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)
        # Cast on meta (free) before allocating: to_empty keeps the dtype, and allocating this 12.9B
        # model at the fp32 default would want ~52GB before the cast ever ran. The norms stay fp32
        # (diffusers' own `_keep_in_fp32_modules`) - casting them costs the fused rms_norm kernel.
        keep_fp32 = tuple(getattr(Krea2Transformer2DModel, "_keep_in_fp32_modules", None) or ())
        model = model.to(dtype)
        for name, param in model.named_parameters():
            if _keeps_fp32(name, keep_fp32):
                param.data = param.data.to(torch.float32)

        # Resolved against module names on the meta model, so a mismatched LoRA is refused before
        # the 26GB read rather than a block into it.
        plan = lora_module.plan_loras(model, loras, alias=convert.module_alias)

        target_device = device or "cpu"
        # NF4 quantizes as bitsandbytes moves a block to the GPU, so those stage on the CPU first.
        # Every other plan streams each tensor straight to its final device.
        staging = "cpu" if quant is Quantization.NF4 else target_device
        expected = {name for name, _ in model.named_parameters()}
        expected |= {name for name, _ in model.named_buffers()}

        with CheckpointReader(file) as handle:
            keys = handle.keys()
            convert.check_loadable(keys, expected)
            sources = {convert.convert_key(key): key for key in keys}
            for prefix, chunk in _krea2_chunks(model):
                chunk.to_empty(device=staging)
                for name, target in _named_tensors(chunk, prefix):
                    source = handle.get_tensor(sources[name], device=staging)
                    target.data.copy_(source.reshape(target.shape).to(target.dtype))
                # Fuse before quantizing: the fuse adds a full-precision delta that quantized
                # weights cannot accept in place.
                lora_module.apply_plan(chunk, plan, f"{prefix}.")
                _quantize_chunk(chunk, quant, target_device)
        return model

    key = (
        "krea2", "diffusion", file, _dtype_key(dtype), quant.value, _device_key(device),
        *lora_cache_key(loras),
    )
    return _cached(key, build)


def _krea2_chunks(model: Any) -> Any:
    """The model in load-sized pieces: each transformer block, then the smaller top-level modules.
    A block is ~0.9GB at bf16, so this is the granularity that bounds peak memory."""
    for index, block in enumerate(model.transformer_blocks):
        yield f"transformer_blocks.{index}", block
    for name, child in model.named_children():
        if name != "transformer_blocks":
            yield name, child


def _named_tensors(module: Any, prefix: str) -> Any:
    """The module's parameters and buffers under their full path in the parent model."""
    for name, tensor in module.named_parameters():
        yield f"{prefix}.{name}", tensor
    for name, tensor in module.named_buffers():
        yield f"{prefix}.{name}", tensor


def _quantize_chunk(chunk: Any, quant: Quantization, device: str) -> None:
    """Shrink one materialized chunk and put it on its final device."""
    if quant is Quantization.NF4:
        _swap_to_4bit(chunk)
        chunk.to(device)  # bitsandbytes quantizes each weight during this move
    else:
        _quantize_in_place(chunk, quant)


def _swap_to_4bit(module: Any) -> None:
    """Replace every ``nn.Linear`` with a bitsandbytes NF4 layer carrying the same weights.

    The QLoRA arrangement: the base is frozen and 4-bit, gradients still flow through it to the
    LoRA adapter on top. Quantization itself happens when the layer moves to CUDA."""
    import bitsandbytes as bnb
    import torch

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
                quant_type="nf4",
            )
            layer.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type="nf4"
            )
            if child.bias is not None:
                layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, layer)
        else:
            _swap_to_4bit(child)


def _keeps_fp32(name: str, keep: tuple[str, ...]) -> bool:
    """Whether a parameter belongs to one of diffusers' keep-in-fp32 modules. Matched per path
    segment so ``norm`` does not also catch ``norm_out`` or ``txt_in.norm_x``."""
    segments = set(name.split("."))
    return any(k in segments for k in keep)


def load_qwen_image_vae(file: str, dtype: Any, device: str | None = None) -> Any:
    """The Qwen-Image VAE from a diffusers-format ``.safetensors`` + the bundled config.

    ComfyUI's ``qwen_image_vae.safetensors`` is a **different layout** (flat
    ``upsamples.N.residual`` sequentials where diffusers has named ``up_blocks.N.resnets.N``
    modules) and diffusers ships no converter, so that file is refused with a pointer at the popup.
    """

    def build() -> Any:
        import torch
        from diffusers import AutoencoderKLQwenImage
        from safetensors.torch import load_file

        root = ensure_assets("krea2")
        config = AutoencoderKLQwenImage.load_config(str(root), subfolder="vae")
        with torch.device("meta"):
            model = AutoencoderKLQwenImage.from_config(config)
        expected = set(model.state_dict())
        state = load_file(file, device=device or "cpu")
        if not expected.issubset(state):
            raise ComponentError(
                "This VAE is the ComfyUI-layout Qwen-Image VAE, which diffusers cannot read. "
                "Download the Krea 2 VAE from the node's model popup (it fetches the "
                "diffusers-format file from Qwen/Qwen-Image)."
            )
        model = model.to_empty(device=device or "cpu")
        model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True, assign=True)
        return model.to(dtype)

    return _cached(
        ("krea2", "vae", file, _dtype_key(dtype), Quantization.NONE.value, _device_key(device)),
        build,
    )


def load_qwen3vl_text_encoder(
    file: str,
    dtype: Any,
    quant: Quantization = Quantization.NONE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """The Qwen3-VL text encoder + tokenizer, staged like Z-Image's so transformers streams tensors
    to the GPU instead of materializing the whole 8.9GB encoder in host RAM."""

    def build() -> tuple[Any, Any]:
        from transformers import AutoTokenizer, Qwen3VLModel

        root = ensure_assets("krea2")
        weights_dir = _staged_encoder_dir("krea2", file)
        text_encoder = Qwen3VLModel.from_pretrained(
            str(weights_dir),
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device else None,
            local_files_only=True,
            quantization_config=_quant_config(quant, framework="transformers"),
        )
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        return text_encoder, tokenizer

    return _cached(
        ("krea2", "text_encoder", file, _dtype_key(dtype), quant.value, _device_key(device)), build
    )


def assemble_krea2_pipeline(
    *,
    diffusion_file: str,
    vae_file: str,
    text_encoder_file: str,
    dtype: Any,
    distilled: bool,
    quant: Quantization = Quantization.NONE,
    vae_dtype: Any = None,
    device: str | None = None,
    loras: tuple[LoraRef, ...] = (),
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """Build a ``Krea2Pipeline`` from three local files. Unplaced - the runner owns placement.
    ``distilled`` is the Turbo checkpoint's fixed ``mu=1.15`` schedule; RAW computes mu from the
    resolution. Buffers are released between the big loads so peak memory tracks one component."""
    from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline

    transformer = load_krea2_transformer(diffusion_file, dtype, quant, device=device, loras=loras)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    vae = load_qwen_image_vae(vae_file, dtype if vae_dtype is None else vae_dtype, device=device)
    _release_transient()
    if cancel_check is not None:
        cancel_check()
    text_encoder, tokenizer = load_qwen3vl_text_encoder(
        text_encoder_file, dtype, quant, device=device
    )
    _release_transient()
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(KREA2_SCHEDULER_CONFIG)
    return Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=distilled,
    )


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
    loras: tuple[LoraRef, ...] = (),
    controlnet_file: str | None = None,
    cancel_check: Callable[[], None] | None = None,
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
    from diffusers import ZImageControlNetPipeline, ZImageImg2ImgPipeline, ZImagePipeline

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
    if controlnet_file:
        # ControlNet runs text-to-image + control (no img2img in this path). The pipeline __init__
        # grafts the shared transformer layers onto the ControlNet via from_transformer.
        controlnet = load_zimage_controlnet(controlnet_file, dtype, device=device)
        _release_transient()
        return ZImageControlNetPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            controlnet=controlnet,
        )
    cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
    return cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
