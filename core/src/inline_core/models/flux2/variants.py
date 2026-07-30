"""Which FLUX.2 checkpoint a file actually is, and what it needs to run.

FLUX.2 ships as one family with two incompatible halves: ``dev`` uses a Mistral-3 text encoder and
``Flux2Pipeline``; every ``klein`` build uses Qwen3 and ``Flux2KleinPipeline``. They also differ in
step count, guidance semantics, and which encoder layers get tapped. One node covers all of them by
identifying the picked file here and looking the rest up.

Identification reads the **safetensors header only** (tensor names and shapes, never tensor data,
never torch), so it is cheap enough to run on every model-popup open and works on a torch-less
install. Deriving the transformer geometry from the checkpoint instead of shipping a config per
variant is what keeps a future FLUX.3 build - or a community fine-tune - loadable with no code
change, and it avoids depending on the gated BFL repos for a config we can already see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "VARIANTS",
    "Flux2Variant",
    "derive_transformer_config",
    "detect",
    "get",
    "text_encoder_kind",
]

#: Geometry shared by every FLUX.2 transformer that tensor shapes cannot reveal. Everything else in
#: the config is derived from the checkpoint (see ``derive_transformer_config``).
_FIXED_GEOMETRY: dict[str, object] = {
    "axes_dims_rope": [32, 32, 32, 32],
    "eps": 1e-06,
    "patch_size": 1,
    "rope_theta": 2000,
    "out_channels": None,
}


@dataclass(frozen=True)
class Flux2Variant:
    """One FLUX.2 checkpoint family: how to build it, and what it wants at sampling time."""

    key: str
    label: str
    #: Which diffusers pipeline family to build: "klein" | "klein-kv" | "dev".
    pipeline: str
    #: Step- and guidance-distilled. Drives the auto sampler defaults and whether real CFG runs.
    distilled: bool
    #: The concatenated text-encoder width (3x the encoder's hidden size). The identifying number.
    joint_attention_dim: int
    #: Which intermediate encoder layers the pipeline stacks. dev taps deeper than klein.
    text_encoder_layers: tuple[int, ...]
    #: The loader arch key - selects the tokenizer/config asset bundle in models/loaders.py.
    arch: str
    steps: int
    guidance: float

    @property
    def supports_negative_prompt(self) -> bool:
        """Only an undistilled klein checkpoint runs real CFG. dev is guidance-distilled and its
        pipeline has no negative path at all; a distilled klein ignores guidance above 1."""
        return not self.distilled and self.pipeline in ("klein", "klein-kv")


#: Every variant the node knows. Adding a checkpoint (one we missed, or FLUX.3) is a row here plus a
#: download entry in requirements.py - no new node, descriptor, or runner branch.
VARIANTS: tuple[Flux2Variant, ...] = (
    Flux2Variant(
        key="klein-4b",
        label="Klein 4B",
        pipeline="klein",
        distilled=True,
        joint_attention_dim=7680,
        text_encoder_layers=(9, 18, 27),
        arch="flux2-klein-4b",
        steps=4,
        guidance=1.0,
    ),
    Flux2Variant(
        key="klein-4b-base",
        label="Klein 4B Base",
        pipeline="klein",
        distilled=False,
        joint_attention_dim=7680,
        text_encoder_layers=(9, 18, 27),
        arch="flux2-klein-4b",
        steps=50,
        guidance=4.0,
    ),
    Flux2Variant(
        key="klein-9b",
        label="Klein 9B",
        pipeline="klein",
        distilled=True,
        joint_attention_dim=12288,
        text_encoder_layers=(9, 18, 27),
        arch="flux2-klein-9b",
        steps=4,
        guidance=1.0,
    ),
    Flux2Variant(
        key="klein-9b-base",
        label="Klein 9B Base",
        pipeline="klein",
        distilled=False,
        joint_attention_dim=12288,
        text_encoder_layers=(9, 18, 27),
        arch="flux2-klein-9b",
        steps=50,
        guidance=4.0,
    ),
    Flux2Variant(
        key="klein-9b-kv",
        label="Klein 9B KV",
        pipeline="klein-kv",
        distilled=True,
        joint_attention_dim=12288,
        text_encoder_layers=(9, 18, 27),
        arch="flux2-klein-9b",
        steps=4,
        guidance=1.0,
    ),
    Flux2Variant(
        key="dev",
        label="dev",
        pipeline="dev",
        distilled=False,
        joint_attention_dim=15360,
        text_encoder_layers=(10, 20, 30),
        arch="flux2-dev",
        steps=28,
        guidance=4.0,
    ),
)

_BY_KEY = {v.key: v for v in VARIANTS}


def get(key: str | None) -> Flux2Variant | None:
    return _BY_KEY.get((key or "").strip())


def text_encoder_kind(variant: Flux2Variant) -> str:
    """Which encoder class the loader builds for this variant."""
    return "mistral3" if variant.pipeline == "dev" else "qwen3"


# --- identifying a checkpoint --------------------------------------------------------------------

#: Prefixes ComfyUI-style repacks put in front of the diffusers keys. Stripped before matching so a
#: Comfy single file and a diffusers export identify the same way.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.")

_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")
_SINGLE_BLOCK_RE = re.compile(r"^single_transformer_blocks\.(\d+)\.")
_WEIGHT_SUFFIXES = (".safetensors", ".sft")

#: The shipped checkpoints use BFL's original key layout, not diffusers'. diffusers converts it at
#: load time (``convert_flux2_transformer_checkpoint_to_diffusers``), but identification runs on the
#: raw header, so the handful of keys we read are renamed here first. Only the identifying keys are
#: mapped - this is not a checkpoint converter, and it must not become one.
_BFL_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^double_blocks\."), "transformer_blocks."),
    (re.compile(r"^single_blocks\."), "single_transformer_blocks."),
    (re.compile(r"^txt_in\.weight$"), "context_embedder.weight"),
    (re.compile(r"^img_in\.weight$"), "x_embedder.weight"),
    (
        re.compile(r"^time_in\.in_layer\.weight$"),
        "time_guidance_embed.timestep_embedder.linear_1.weight",
    ),
    (re.compile(r"(img_)?attn\.norm\.query_norm\.scale$"), "attn.norm_q.weight"),
    (re.compile(r"(?<=\.)norm\.query_norm\.scale$"), "attn.norm_q.weight"),
    (re.compile(r"img_mlp\.0\.weight$"), "ff.linear_in.weight"),
)


def _strip(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    for pattern, replacement in _BFL_RENAMES:
        key = pattern.sub(replacement, key)
    return key


def derive_transformer_config(shapes: dict[str, list[int]]) -> dict[str, object] | None:
    """Reconstruct a ``Flux2Transformer2DModel`` config from a checkpoint's tensor shapes.

    Returns None when the file is not a FLUX.2 transformer. The stream widths come from
    ``context_embedder``/``x_embedder``, the head size from a per-head QK-norm vector, and the block
    counts from the indices present, so the same code sizes a 4B and a 32B checkpoint.
    """
    keys = {_strip(key): shape for key, shape in shapes.items()}
    context = keys.get("context_embedder.weight")
    x_embed = keys.get("x_embedder.weight")
    if not context or not x_embed or len(context) != 2 or len(x_embed) != 2:
        return None

    inner_dim, joint_attention_dim = context[0], context[1]
    head_dim = next(
        (shape[0] for key, shape in keys.items() if key.endswith("attn.norm_q.weight") and shape),
        0,
    )
    if not head_dim or inner_dim % head_dim:
        return None

    layers = {int(m.group(1)) for key in keys if (m := _BLOCK_RE.match(key))}
    single_layers = {int(m.group(1)) for key in keys if (m := _SINGLE_BLOCK_RE.match(key))}
    if not layers or not single_layers:
        return None

    # SwiGLU packs gate and value into one projection, so linear_in is 2 x mlp_ratio x inner_dim.
    ff_in = next(
        (shape[0] for key, shape in keys.items() if key.endswith("ff.linear_in.weight") and shape),
        0,
    )
    time_in = keys.get("time_guidance_embed.timestep_embedder.linear_1.weight") or []

    return {
        **_FIXED_GEOMETRY,
        "attention_head_dim": head_dim,
        "guidance_embeds": any(
            "guidance_embedder" in key or key.startswith("guidance_in.") for key in keys
        ),
        "in_channels": x_embed[1],
        "joint_attention_dim": joint_attention_dim,
        "mlp_ratio": round(ff_in / (2 * inner_dim), 4) if ff_in else 3.0,
        "num_attention_heads": inner_dim // head_dim,
        "num_layers": len(layers),
        "num_single_layers": len(single_layers),
        "timestep_guidance_channels": time_in[1] if len(time_in) == 2 else 256,
    }


def _name_flags(name: str) -> tuple[bool, bool]:
    """(is_base, is_kv) read from a filename. Base and distilled builds are architecturally
    identical, so the name is the only signal - BFL and every repacker mark it the same way."""
    padded = "-" + re.sub(r"[^a-z0-9]+", "-", name.lower()) + "-"
    return "-base-" in padded, "-kv-" in padded


def detect(path: str | Path, shapes: dict[str, list[int]] | None = None) -> Flux2Variant | None:
    """Which FLUX.2 variant a checkpoint file is, or None if it is not one.

    Pass ``shapes`` when the header has already been read, so it is not read twice. Never reads
    tensor data and never imports torch.
    """
    file = Path(path)
    if shapes is None:
        if not file.is_file() or file.suffix.lower() not in _WEIGHT_SUFFIXES:
            return None
        try:
            from ..checkpoint import CheckpointReader

            shapes = CheckpointReader(file).shapes()
        except Exception:  # noqa: BLE001 - an unreadable or foreign file is simply "not FLUX.2"
            return None

    config = derive_transformer_config(shapes)
    if config is None:
        return None
    is_base, is_kv = _name_flags(file.name)
    family = [v for v in VARIANTS if v.joint_attention_dim == config["joint_attention_dim"]]
    if not family:
        return None
    # dev has no base/distilled or KV split, so those name flags must not filter it away.
    exact = [
        v
        for v in family
        if v.distilled is not is_base and (v.pipeline == "klein-kv") is (is_kv and v.distilled)
    ]
    return (exact or family)[0]
