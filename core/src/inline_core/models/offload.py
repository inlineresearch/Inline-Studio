"""Turn the device policy's verdict into a concrete quantisation and offload recipe.

The policy owns *what* plan a model gets ("resident", "int8", "offload", "wont-fit"). This module
owns *how* that plan is carried out for a model too large to simply place, which is the case
diffusers' ``enable_model_cpu_offload`` does not cover: a 60 GB transformer needs its blocks
streamed from host RAM while it runs, not swapped whole.

Nothing here picks a device. It reads the plan the policy already chose, which is the rule that
keeps one graph portable across a 4090, a laptop and a CPU box.

Torch-free at import, like ``models/sampling.py``'s data layer: the recipe is a dataclass anyone can
inspect, and torch/torchao are imported lazily inside ``apply_offload``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..device.policy import DevicePolicy, Quantization

logger = logging.getLogger("inline_core.offload")

#: Group-offload granularities, in the order they cost speed. ``block_level`` moves whole
#: transformer blocks; ``leaf_level`` moves individual modules and fits smaller cards.
BLOCK_LEVEL = "block_level"
LEAF_LEVEL = "leaf_level"


@dataclass(frozen=True)
class OffloadRecipe:
    """How to place one oversized model. Every field is inspectable so a test can assert the map
    without a GPU, and so the log line can name what actually happened."""

    plan: str
    quantize: Quantization = Quantization.NONE
    #: Modules left in high precision. For MiniMax H3 this list was read off the published int8
    #: build rather than guessed: only the attention projections, the FFN and the AdaLN projection
    #: were quantised there.
    keep_precision: tuple[str, ...] = ()
    denoiser_offload: str | None = None
    #: Streamed offload needs pinnable weights, which is why the int8 path asks for version 2
    #: tensors; without them this has to be False or the copies serialise against compute.
    use_stream: bool = False
    encoder_offload: str | None = None
    vae_offload: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def quantizes(self) -> bool:
        return self.quantize is not Quantization.NONE


def recipe_for(
    policy: DevicePolicy,
    *,
    keep_precision: tuple[str, ...] = (),
    stream_denoiser: bool = False,
) -> OffloadRecipe:
    """The recipe for whatever plan the policy landed on for the current footprint.

    ``stream_denoiser`` is for a model whose conditioner has to be co-resident and is large enough
    that the pair will not fit even quantised. The fit ladder in ``core/CLAUDE.md`` sizes the
    denoiser alone, which is right for image models where the text encoder is a rounding error and
    wrong for MiniMax H3, whose conditioner is a 32B model in its own right.
    """
    fit = policy.fit_estimate()
    plan = fit.plan if fit is not None else "resident"
    if plan == "wont-fit":
        # The runner is expected to have refused already; carry the verdict rather than invent one.
        return OffloadRecipe(plan=plan, notes=("model does not fit this machine",))
    if plan == "resident":
        return OffloadRecipe(plan=plan, notes=("weights stay on the accelerator",))
    if plan == "int8":
        # int8 means **resident** by default, per the ladder in core/CLAUDE.md: quantisation is what
        # buys the fit. A caller whose conditioner must sit beside it says so, and then the denoiser
        # streams instead, trading step time for the VRAM the conditioner needs.
        return OffloadRecipe(
            plan=plan,
            quantize=Quantization.INT8,
            keep_precision=keep_precision,
            denoiser_offload=BLOCK_LEVEL if stream_denoiser else None,
            use_stream=stream_denoiser,
            vae_offload=LEAF_LEVEL,
            notes=(
                "int8 weights, denoiser blocks streamed from host RAM"
                if stream_denoiser
                else "int8 weights resident on the accelerator",
            ),
        )
    # "offload": too tight even for streamed int8 blocks, so the VAE joins them and granularity
    # drops to leaf level. Slower per step, but it is the difference between running and not.
    return OffloadRecipe(
        plan=plan,
        quantize=Quantization.INT8,
        keep_precision=keep_precision,
        denoiser_offload=LEAF_LEVEL,
        use_stream=False,
        encoder_offload=LEAF_LEVEL,
        vae_offload=LEAF_LEVEL,
        notes=("int8 weights, everything streamed leaf-by-leaf", "expect a slow render"),
    )


def quantization_config(recipe: OffloadRecipe, *, for_transformers: bool = False) -> Any:
    """A diffusers or transformers ``TorchAoConfig`` for load-time quantisation, or None.

    ``version=2`` matters: those int8 tensors are pinnable, and pinned memory is what streamed
    offload needs to overlap its copies with compute.
    """
    if not recipe.quantizes:
        return None
    try:
        from torchao.quantization import Int8WeightOnlyConfig
    except ImportError:
        logger.warning("torchao is not installed; loading in full precision instead of int8.")
        return None
    if for_transformers:
        from transformers import TorchAoConfig as Config
    else:
        from diffusers import TorchAoConfig as Config
    return Config(
        Int8WeightOnlyConfig(version=2), modules_to_not_convert=list(recipe.keep_precision)
    )


def encoder_quantization_config(recipe: OffloadRecipe) -> Any:
    """A 4-bit config for a conditioner that is far larger than the model it conditions.

    int8 is the right rung for a denoiser, whose weights are read every step of every render. A
    text encoder runs once per prompt and is then idle, so it can afford the heavier compression:
    NF4 takes MiniMax H3's Qwen3-VL-32B from 66.7 GB to roughly 19 GB, which is what makes the pair
    fit beside each other at all. This is bitsandbytes' own format, not a repacked one, so nothing
    here depends on decoding another framework's scale tensors.
    """
    if not recipe.quantizes:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("bitsandbytes is not installed; the text encoder stays in full precision.")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Quantising the quantisation constants too; the encoder is idle during denoise, so the
        # extra unpack cost is paid once per prompt rather than once per step.
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=list(recipe.keep_precision) or None,
    )


def apply_offload(
    recipe: OffloadRecipe,
    *,
    denoiser: Any = None,
    encoder: Any = None,
    vae: Any = None,
    device: Any = None,
) -> None:
    """Install the recipe's group-offload hooks on whichever components were handed over."""
    if not any((recipe.denoiser_offload, recipe.encoder_offload, recipe.vae_offload)):
        return
    import torch
    from diffusers.hooks import apply_group_offloading

    onload = torch.device(str(device)) if device is not None else torch.device("cuda")
    common = {"onload_device": onload, "offload_device": torch.device("cpu")}
    for module, granularity, stream in (
        (denoiser, recipe.denoiser_offload, recipe.use_stream),
        (encoder, recipe.encoder_offload, recipe.use_stream),
        (vae, recipe.vae_offload, False),
    ):
        if module is None or granularity is None:
            continue
        # Quantised tensors cannot serve autograd, and nothing here trains, so freeze before the
        # hooks go on rather than discovering it mid-step.
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
        options: dict[str, Any] = {**common, "offload_type": granularity, "use_stream": stream}
        if granularity == BLOCK_LEVEL:
            options["num_blocks_per_group"] = 1
        apply_group_offloading(module, **options)
    logger.info("Offload recipe applied: %s", describe(recipe))


def describe(recipe: OffloadRecipe) -> str:
    """A one-line summary for the run log, so a slow render has a visible reason."""
    parts = [f"plan={recipe.plan}"]
    if recipe.quantizes:
        parts.append(f"quant={recipe.quantize.value}")
    for name, value in (
        ("denoiser", recipe.denoiser_offload),
        ("encoder", recipe.encoder_offload),
        ("vae", recipe.vae_offload),
    ):
        if value:
            parts.append(f"{name}={value}" + ("+stream" if recipe.use_stream and value else ""))
    parts.extend(recipe.notes)
    return ", ".join(parts)
