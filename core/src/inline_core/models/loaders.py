"""Model-agnostic single-file loaders — assemble a diffusers pipeline from individual weight files
(ComfyUI-style), never touching the network at generation time.

The problem this solves: loading a whole diffusers *pipeline folder* with ``from_pretrained`` is
slow (sharded weights reconstructed from an index, per-component metadata resolution, and a stall if
any config is not local). ComfyUI is fast because it loads **one consolidated ``.safetensors`` per
component**, builds each model from a config it ships, and bundles the tokenizer — zero hub
round-trip. This module does the same:

  - the big weights come from the user's three files (``diffusion_models/`` / ``vae/`` /
    ``text_encoders/``), loaded via ``from_single_file`` / a config-driven ``state_dict`` load;
  - the small configs + the Qwen tokenizer come from a **fetch-once** asset bundle
    (``ensure_assets``), pulled once into the engine data dir and then reused offline.

An ``ArchSpec`` maps a model family to its reference repo (for the small assets) and to how each
component is built. Only ``z-image`` is wired today; Flux and friends slot in as new specs — the
Z-Image node's dropdowns and the decomposed ``load/*`` subnodes both call through here, so a new
arch is a data change, not new plumbing.

Heavy deps (torch, diffusers, transformers, huggingface_hub, safetensors) are imported **inside**
functions on purpose: importing this module stays cheap, and a torch-less engine boot never trips
over it. Callers are the model-runner subpackages, which the server registers best-effort.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import data_dir
from ..device.policy import Quantization
from ..errors import ComponentError


@dataclass(frozen=True)
class ArchSpec:
    """How to source the small offline assets (configs + tokenizer) for one model family.

    ``asset_files`` are repo-relative paths under ``assets_repo`` — the tiny config/tokenizer files
    (never the multi-GB weights) fetched once into ``data_dir()/assets/<key>``. The subfolder layout
    is preserved, so a component loads its config with ``config=<assets_root>, subfolder="<name>"``.
    """

    key: str
    assets_repo: str
    asset_files: tuple[str, ...]


#: The reference repo carries the small diffusers configs + the Qwen3 tokenizer. Weights are never
#: taken from here — the user supplies those as single files (see requirements.py).
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

    Raises ComponentError if the first fetch can't reach the hub — the assets are the only piece not
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
        except Exception as error:  # noqa: BLE001 — re-raised as a clear component error
            raise ComponentError(
                f"Could not fetch the one-time {arch} config/tokenizer assets from "
                f"{spec.assets_repo} ({error}). This ~15 MB download happens once; after it, "
                "generation runs fully offline."
            ) from error
        marker.write_text("ok")
    return root


# --- quantization ------------------------------------------------------------------------------


def _quant_config(quant: Quantization, *, framework: str) -> Any | None:
    """A diffusers/transformers ``quantization_config`` for the requested weight quantization, or
    None for full precision. ``framework`` picks which library's config class to build:
    ``"diffusers"`` for the transformer, ``"transformers"`` for the Qwen3 text encoder.

    INT8 uses torch-native weight-only quantization (torchao) on purpose: unlike bitsandbytes it
    stays a movable tensor subclass, so it coexists with ``enable_model_cpu_offload`` — exactly the
    smart-memory case that spreads the model across VRAM + RAM. NF4 (bitsandbytes) is CUDA-only.
    Best-effort: if the optional backend (torchao/bitsandbytes) is not installed we return None and
    load full precision rather than crash — the caller (smart memory) still gets CPU offload."""
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
    except Exception:  # noqa: BLE001 — a missing/incompatible quant backend must not break loading
        import logging

        logging.getLogger("inline_core.zimage").warning(
            "Quantization %s (%s) unavailable — loading full precision. Install torchao (int8) or "
            "bitsandbytes (nf4) to shrink weights for smart memory.",
            quant.value,
            framework,
        )
        return None


# --- component loaders (cached in-process) ------------------------------------------------------

# Keyed by (arch, kind, file, dtype, quant) so switching one file (e.g. a different VAE) or the
# quantization reuses the other already-loaded components. The run manager executes one run at a
# time; the lock guards each build.
_CACHE: dict[tuple[str, str, str, str, str], Any] = {}
_CACHE_LOCK = Lock()


def _cached(key: tuple[str, str, str, str, str], build: Callable[[], Any]) -> Any:
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        value = build()
        _CACHE[key] = value
        return value


def _dtype_key(dtype: Any) -> str:
    return str(dtype)


def load_diffusion(
    arch: str, file: str, dtype: Any, quant: Quantization = Quantization.NONE
) -> Any:
    """The diffusion transformer from a single ``.safetensors``. diffusers converts the checkpoint
    keys; the config comes from the bundled assets, so nothing is fetched at load time. ``quant``
    (smart memory) quantizes the weights on load so the model fits across VRAM + RAM."""

    def build() -> Any:
        from diffusers import ZImageTransformer2DModel

        root = ensure_assets(arch)
        return ZImageTransformer2DModel.from_single_file(
            file,
            config=str(root),
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=True,
            quantization_config=_quant_config(quant, framework="diffusers"),
        )

    return _cached((arch, "diffusion", file, _dtype_key(dtype), quant.value), build)


def load_vae(arch: str, file: str, dtype: Any) -> Any:
    """The VAE from a single ``.safetensors`` (the Flux/LDM-style ``ae.safetensors``). diffusers'
    LDM-VAE converter remaps the keys; the config is the bundled ``AutoencoderKL`` config."""

    def build() -> Any:
        from diffusers import AutoencoderKL

        root = ensure_assets(arch)
        return AutoencoderKL.from_single_file(
            file,
            config=str(root),
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=True,
        )

    # The VAE stays full precision (it is small — a few hundred MB — and int8 on the conv VAE costs
    # quality for no meaningful memory win); the key still carries a quant slot for a uniform shape.
    return _cached((arch, "vae", file, _dtype_key(dtype), Quantization.NONE.value), build)


def load_text_encoder(
    arch: str, file: str, dtype: Any, quant: Quantization = Quantization.NONE
) -> tuple[Any, Any]:
    """The Qwen3 text encoder + tokenizer. The weights come from the user's single file; the config
    and tokenizer come from the bundled assets. We load the state dict directly (``from_pretrained``
    with an explicit ``state_dict`` — no 16 GB fp32 init), and transformers strips the ``model.``
    prefix automatically (Qwen3Model's ``base_model_prefix`` is ``"model"``). ``quant`` (smart
    memory) quantizes the encoder — the largest weight — so the offloaded model fits in RAM."""

    def build() -> tuple[Any, Any]:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

        root = ensure_assets(arch)
        te_dir = root / "text_encoder"
        config = Qwen3Config.from_pretrained(str(te_dir), local_files_only=True)
        state_dict = load_file(file)
        text_encoder = Qwen3Model.from_pretrained(
            None,
            config=config,
            state_dict=state_dict,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            quantization_config=_quant_config(quant, framework="transformers"),
        )
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        return text_encoder, tokenizer

    return _cached((arch, "text_encoder", file, _dtype_key(dtype), quant.value), build)


def load_scheduler(arch: str) -> Any:
    """The flow-match scheduler, from the bundled config (config-only — never downloads)."""
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
) -> Any:
    """Build a Z-Image pipeline from three local single files. Components are cached individually,
    so swapping one file reuses the others. The returned pipeline is unplaced — the runner owns
    device placement / low-VRAM tweaks. ``quant`` (smart memory) quantizes the big weights (the
    transformer + text encoder) on load. ``vae_dtype`` (defaults to ``dtype``) lets the VAE keep a
    safer dtype than the denoiser — e.g. fp32 when the transformer runs fp16 (whose decode can
    overflow)."""
    from diffusers import ZImageImg2ImgPipeline, ZImagePipeline

    arch = _ZIMAGE.key
    transformer = load_diffusion(arch, diffusion_file, dtype, quant)
    vae = load_vae(arch, vae_file, dtype if vae_dtype is None else vae_dtype)
    text_encoder, tokenizer = load_text_encoder(arch, text_encoder_file, dtype, quant)
    scheduler = load_scheduler(arch)
    cls = ZImageImg2ImgPipeline if img2img else ZImagePipeline
    return cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
