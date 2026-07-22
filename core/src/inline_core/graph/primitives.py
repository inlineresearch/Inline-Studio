"""The low-level primitive node vocabulary (our own clean design, not Comfy's names).

Descriptors only here; the diffusers-backed runners land in C2. Registering them descriptor-first
lets /v1/models serve the palette and the validator type-check engine-typed edges (model, vae,
conditioning, latent) now. Only media-output nodes (vae/decode) become Frames with take history.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..media import MediaKind
from .descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from .schema import PortKind

if TYPE_CHECKING:
    from .registry import Registry

_SAMPLERS = (Option("euler", "Euler"), Option("dpmpp_2m", "DPM++ 2M"), Option("heun", "Heun"))
_SCHEDULERS = (Option("simple", "Simple"), Option("karras", "Karras"))

LOAD_DIFFUSION_MODEL = NodeDescriptor(
    type="load/diffusion-model",
    title="Load Diffusion Model",
    category="Loaders",
    icon="box",
    params=(ParamField("file", "Model", Widget.SELECT, "", options_from="diffusion_models"),),
    outputs=(Port("model", "Model", PortKind.MODEL),),
)

LOAD_VAE = NodeDescriptor(
    type="load/vae",
    title="Load VAE",
    category="Loaders",
    icon="box",
    params=(ParamField("file", "VAE", Widget.SELECT, "", options_from="vae"),),
    outputs=(Port("vae", "VAE", PortKind.VAE),),
)

LOAD_TEXT_ENCODER = NodeDescriptor(
    type="load/text-encoder",
    title="Load Text Encoder",
    category="Loaders",
    icon="box",
    params=(ParamField("file", "Text encoder", Widget.SELECT, "", options_from="text_encoders"),),
    outputs=(Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER),),
)

LOAD_LORA = NodeDescriptor(
    type="load/lora",
    title="Load LoRA",
    category="Loaders",
    icon="box",
    # Optional upstream lora: chain load/lora -> load/lora to stack several in order.
    inputs=(Port("lora", "LoRA", PortKind.LORA, required=False),),
    params=(
        ParamField("file", "LoRA", Widget.SELECT, "", options_from="loras"),
        ParamField("strength", "Strength", Widget.NUMBER, 1.0, min=-2.0, max=2.0, step=0.05),
    ),
    outputs=(Port("lora", "LoRA", PortKind.LORA),),
)

ENCODE_TEXT = NodeDescriptor(
    type="encode/text",
    title="Encode Text",
    category="Conditioning",
    icon="type",
    inputs=(
        Port("text_encoder", "Text encoder", PortKind.TEXT_ENCODER, required=True),
        Port("prompt", "Prompt", PortKind.TEXT, required=True),
    ),
    outputs=(Port("conditioning", "Conditioning", PortKind.CONDITIONING),),
)

EMPTY_LATENT = NodeDescriptor(
    type="latent/empty",
    title="Empty Latent",
    category="Latent",
    icon="square",
    params=(
        ParamField("width", "Width", Widget.NUMBER, 1024, min=64, max=4096, step=8),
        ParamField("height", "Height", Widget.NUMBER, 1024, min=64, max=4096, step=8),
        ParamField("batch", "Batch", Widget.NUMBER, 1, min=1, max=16, step=1),
    ),
    outputs=(Port("latent", "Latent", PortKind.LATENT),),
)

SAMPLE = NodeDescriptor(
    type="sample",
    title="Sample",
    category="Sampling",
    icon="wand",
    inputs=(
        Port("model", "Model", PortKind.MODEL, required=True),
        Port("positive", "Positive", PortKind.CONDITIONING, required=True),
        Port("negative", "Negative", PortKind.CONDITIONING, required=False),
        Port("latent", "Latent", PortKind.LATENT, required=True),
    ),
    params=(
        ParamField("steps", "Steps", Widget.NUMBER, 20, min=1, max=200, step=1),
        ParamField("cfg", "CFG", Widget.NUMBER, 5.0, min=0.0, max=30.0, step=0.1),
        ParamField("sampler", "Sampler", Widget.SELECT, "euler", options=_SAMPLERS),
        ParamField("scheduler", "Scheduler", Widget.SELECT, "simple", options=_SCHEDULERS),
        ParamField("seed", "Seed (-1 = random)", Widget.SEED, -1),
    ),
    outputs=(Port("latent", "Latent", PortKind.LATENT),),
)

VAE_DECODE = NodeDescriptor(
    type="vae/decode",
    title="VAE Decode",
    category="VAE",
    icon="image",
    output_kind=MediaKind.IMAGE,
    inputs=(
        Port("vae", "VAE", PortKind.VAE, required=True),
        Port("latent", "Latent", PortKind.LATENT, required=True),
    ),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
)

VAE_ENCODE = NodeDescriptor(
    type="vae/encode",
    title="VAE Encode",
    category="VAE",
    icon="square",
    inputs=(
        Port("vae", "VAE", PortKind.VAE, required=True),
        Port("image", "Image", PortKind.IMAGE, required=True),
    ),
    outputs=(Port("latent", "Latent", PortKind.LATENT),),
)

PRIMITIVES: tuple[NodeDescriptor, ...] = (
    LOAD_DIFFUSION_MODEL,
    LOAD_VAE,
    LOAD_TEXT_ENCODER,
    LOAD_LORA,
    ENCODE_TEXT,
    EMPTY_LATENT,
    SAMPLE,
    VAE_DECODE,
    VAE_ENCODE,
)


#: The loader primitives now have runners and are offered in the add-node menu - registered
#: (unhidden) with their runners by ``graph/loader_runners.py``, so they are skipped here.
_HAS_RUNNER = {LOAD_DIFFUSION_MODEL.type, LOAD_VAE.type, LOAD_TEXT_ENCODER.type, LOAD_LORA.type}


def register_primitives(registry: Registry) -> None:
    """Register the still-descriptor-only primitives, marked hidden so they never surface in the
    add-node menu (they stay available for validation). The loader primitives are registered with
    runners + unhidden by ``register_loaders``; the sampler/encode/decode runners land in C2."""
    for descriptor in PRIMITIVES:
        if descriptor.type in _HAS_RUNNER:
            continue
        registry.register(replace(descriptor, hidden=True))
