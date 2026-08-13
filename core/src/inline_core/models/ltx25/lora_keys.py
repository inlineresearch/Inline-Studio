"""Write an LTX-2.5 adapter in the names and the scaling other LTX tools actually read.

Two things separate what PEFT produces from what the ecosystem expects, and only one of them is a
rename.

**The prefix.** Published LTX LoRAs key every tensor as ``diffusion_model.<module>.lora_A.weight``.
Module paths below it already match ours exactly - the published distilled LoRA targets 1660
modules and the transformer has exactly 1660 Linears, with no mismatch - so the rename is the
prefix and nothing else.

**The scale, which is the part that bites.** PEFT trains with a scale of ``alpha / rank`` and saves
the factors raw, expecting the loader to reapply it. Inline's loader does, from the ``.alpha``
written beside each pair. `ltx_core`'s does **not**: ``_products_for_sd_key`` pairs ``lora_A`` with
``lora_B`` by exact key and applies only the user's strength, ignoring alpha. Upstream never notices
because their published adapters ship ``rank == alpha``, so the scale is 1.0.

So an adapter trained at ``alpha != rank`` would load correctly here and at the wrong strength
everywhere else - silently, because a LoRA at the wrong scale is not an error, just a worse result.
Folding the scale into ``lora_B`` and dropping ``.alpha`` makes the file mean the same thing to
every loader, which is what portability has to mean.
"""

from __future__ import annotations

from typing import Any

#: Every published LTX LoRA carries it; our loader already strips it on the way back in.
PREFIX = "diffusion_model."

_DOWN = ".lora_A.weight"
_UP = ".lora_B.weight"
_ALPHA = ".alpha"


def export_reference(state: dict[str, Any]) -> dict[str, Any]:
    """PEFT's state dict as a published-convention LTX LoRA.

    Takes the ``.alpha`` scalars ``trainer._save_lora`` has already written, folds ``alpha / rank``
    into the up-projection, drops them, and prefixes what remains.
    """
    scales = _scales(state)
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key.endswith(_ALPHA):
            continue  # folded into lora_B below; keeping it would double-apply in our own loader
        if key.endswith(_UP):
            scale = scales.get(key[: -len(_UP)], 1.0)
            value = value * scale if scale != 1.0 else value
        out[f"{PREFIX}{key}"] = value
    return out


def _scales(state: dict[str, Any]) -> dict[str, float]:
    """``module -> alpha / rank``, from the alpha scalars and the rank of each down-projection.

    Rank is read off ``lora_A``'s own first dimension rather than taken from the run's config, so a
    resumed or hand-edited adapter is described by what it contains.
    """
    scales: dict[str, float] = {}
    for key, alpha in state.items():
        if not key.endswith(_ALPHA):
            continue
        module = key[: -len(_ALPHA)]
        down = state.get(f"{module}{_DOWN}")
        rank = int(getattr(down, "shape", (0,))[0]) if down is not None else 0
        if rank:
            scales[module] = float(alpha) / rank
    return scales


def is_reference(state: dict[str, Any]) -> bool:
    """Whether a state dict is already in the published convention."""
    return any(key.startswith(PREFIX) for key in state)
