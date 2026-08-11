"""Translate MiniMax H3 LoRAs between our diffusers module names and the reference's fused ones.

An adapter trained here attaches to the diffusers port's modules
(``transformer_blocks.0.attn.to_q``), while every other tool keys off the released checkpoint's own
names (``blocks.0.attn.qkv_proj``). The two disagree by the same transforms ``keys.py`` already
declares for the base weights, so this is the LoRA-shaped view of that one map rather than a second
copy of it that can drift.

Two facts make the round trip exact:

* Every transform acts on **output rows**, so it rewrites ``lora_B`` and leaves ``lora_A`` alone.
  The fused QKV is the exception, because three modules here are one there.
* Fusing q, k and v cannot keep rank ``r``: each has its own ``A``. Stacking the three ``A``
  matrices and making ``B`` block-diagonal gives the identical delta at rank ``3r``, and the alpha
  triples with it so ``alpha / rank`` is unchanged. Missing that last part divides the adapter by
  three, silently.

The unresolvable half is stated on ``import_reference``: fused row order differs by publisher, and
an adapter carries no base weights to measure it against.
"""

from __future__ import annotations

from typing import Any

from ...errors import ComponentError
from ..keymap import RowLayout, deinterleave_rows, interleave_rows
from ..lora import split_key
from .keys import HEAD_DIM, SOURCE_LAYOUTS

#: Reference stem -> ours, for the block Linears an adapter can attach to. The norms in
#: ``keys._BLOCK_RENAMES`` are absent on purpose: they are not Linear and never carry a LoRA.
_RENAMES = {"attn.out_proj": "attn.to_out.0", "mlp.fc2": "ff.net.2"}

#: Same, but the two halves are exchanged, for the ``SwiGLU`` reason ``keys.py`` gives.
_SWAPPED = {"mlp.fc1": "ff.net.0.proj"}

#: The fused stem and the three it becomes, in row order.
_QKV = "attn.qkv_proj"
_QKV_PARTS = ("attn.to_q", "attn.to_k", "attn.to_v")

#: Linears outside any block. ``keys.py`` maps more; these are the only ones
#: ``training/arch._MINIMAX_H3_TARGETS`` lets an adapter reach.
_TOP_LEVEL = {"condition_proj": "context_embedder"}

#: Block prefixes, longest first so the refiner is not eaten by the bare ``blocks.`` rule.
_PREFIXES = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("blocks.", "transformer_blocks."),
)

#: Stripped from an incoming key. ComfyUI writes the first, PEFT the third.
_WRAPPERS = ("diffusion_model.", "transformer.", "base_model.model.")

_UNSWAP = {v: k for k, v in _SWAPPED.items()}
_UNRENAME = {v: k for k, v in _RENAMES.items()}
_UNTOP = {v: k for k, v in _TOP_LEVEL.items()}


def export_reference(state: dict[str, Any], *, target: str = "comfy-org") -> dict[str, Any]:
    """Our adapter in the reference's fused key names, so other tools can load it."""
    layout = _layout(target)
    grouped = _group(state)
    out: dict[str, Any] = {}
    for stem in sorted(grouped):
        parsed = _parse(stem, reference=False)
        if parsed is None:
            if (name := _UNTOP.get(stem)) is not None:
                _emit(out, f"diffusion_model.{name}", grouped[stem])
            continue
        prefix, index, tail = parsed
        block = f"{prefix[0]}{index}."
        if tail in _QKV_PARTS:
            # Only q drives the fuse; k and v come with it and are skipped when their turn comes.
            if tail == _QKV_PARTS[0]:
                fused = _fuse_qkv(f"{prefix[1]}{index}.", grouped, layout)
                _emit(out, f"diffusion_model.{block}{_QKV}", fused)
        elif (name := _UNSWAP.get(tail)) is not None:
            _emit(out, f"diffusion_model.{block}{name}", _with_swapped_up(grouped[stem]))
        elif (name := _UNRENAME.get(tail)) is not None:
            _emit(out, f"diffusion_model.{block}{name}", grouped[stem])
    if not out:
        raise ComponentError("Nothing in this adapter maps onto MiniMax H3's key names.")
    return out


def import_reference(state: dict[str, Any], *, source: str = "comfy-org") -> dict[str, Any]:
    """A third-party H3 adapter rewritten onto the diffusers port's module names.

    ``source`` picks the fused QKV row order. It cannot be measured here:
    ``keymap.detect_row_layout`` works off a checkpoint's real weights and an adapter has none, so a
    wrong guess renders a plausible wrong video rather than failing. The default is the layout
    published H3 LoRAs are overwhelmingly trained against; the caller names the other explicitly.
    """
    layout = _layout(source)
    grouped = _group(state)
    out: dict[str, Any] = {}
    for stem in sorted(grouped):
        parsed = _parse(stem, reference=True)
        if parsed is None:
            if (name := _TOP_LEVEL.get(stem)) is not None:
                _emit(out, name, grouped[stem])
            continue
        prefix, index, tail = parsed
        block = f"{prefix[1]}{index}."
        if tail == _QKV:
            for part, tensors in _split_qkv(block, grouped[stem], layout).items():
                _emit(out, part, tensors)
        elif (name := _SWAPPED.get(tail)) is not None:
            _emit(out, f"{block}{name}", _with_swapped_up(grouped[stem]))
        elif (name := _RENAMES.get(tail)) is not None:
            _emit(out, f"{block}{name}", grouped[stem])
    if not out:
        raise ComponentError(
            "No MiniMax H3 layers found in this adapter; it was trained for a different model."
        )
    return out


def adapt(state: dict[str, Any], *, source: str = "comfy-org") -> dict[str, Any]:
    """Bring any H3 adapter onto our module names: translate a reference-keyed one, pass ours on."""
    if not is_reference(state):
        return state
    return import_reference(state, source=source)


def is_reference(state: dict[str, Any]) -> bool:
    """Whether this adapter is keyed to the released checkpoint rather than the diffusers port.

    Checked on the prefix, not on ``qkv_proj``: an adapter that only touched the feed-forward has no
    fused tensor to give it away."""
    for key in state:
        if (split := split_key(key)) is None:
            continue
        stem = split[0]
        for wrapper in _WRAPPERS:
            stem = stem.removeprefix(wrapper)
        if stem in _TOP_LEVEL or any(stem.startswith(pair[0]) for pair in _PREFIXES):
            return True
    return False


def _layout(name: str) -> RowLayout:
    try:
        return SOURCE_LAYOUTS[name]
    except KeyError:
        raise ComponentError(
            f"Unknown MiniMax H3 LoRA layout {name!r}; expected one of {sorted(SOURCE_LAYOUTS)}."
        ) from None


def _parse(stem: str, *, reference: bool) -> tuple[tuple[str, str], str, str] | None:
    """``(prefix pair, block index, module tail)``, or None when the stem is not inside a block."""
    for pair in _PREFIXES:
        prefix = pair[0] if reference else pair[1]
        if stem.startswith(prefix):
            index, _, tail = stem[len(prefix) :].partition(".")
            return pair, index, tail
    return None


def _group(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """LoRA tensors bucketed by module stem, with the wrapper prefixes stripped."""
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if (split := split_key(key)) is None:
            continue
        stem, role = split
        for wrapper in _WRAPPERS:
            if stem.startswith(wrapper):
                stem = stem[len(wrapper) :]
        grouped.setdefault(stem, {})[role] = value
    return grouped


def _emit(out: dict[str, Any], name: str, tensors: dict[str, Any]) -> None:
    for role, suffix in (("down", "lora_A.weight"), ("up", "lora_B.weight"), ("alpha", "alpha")):
        if role in tensors:
            out[f"{name}.{suffix}" if role != "alpha" else f"{name}.alpha"] = tensors[role]


def _with_swapped_up(tensors: dict[str, Any]) -> dict[str, Any]:
    return {**tensors, "up": _swap_halves(tensors["up"])}


def _fuse_qkv(
    block: str, grouped: dict[str, dict[str, Any]], layout: RowLayout
) -> dict[str, Any]:
    """``[Bq@Aq; Bk@Ak; Bv@Av]`` as one rank-``3r`` adapter: stacked ``A``, block-diagonal ``B``."""
    import torch

    parts = [grouped.get(f"{block}{part}") for part in _QKV_PARTS]
    present = [p for p in parts if p is not None]
    if len(present) != len(_QKV_PARTS):
        raise ComponentError(
            f"{block}attn needs q, k and v adapted together to fuse into one qkv_proj, but "
            f"{len(present)} of 3 are in this adapter."
        )
    up: Any = torch.block_diag(*(p["up"] for p in present))
    if layout is RowLayout.INTERLEAVED:
        up = interleave_rows(up, len(_QKV_PARTS), HEAD_DIM)
    tensors: dict[str, Any] = {"down": torch.cat([p["down"] for p in present], dim=0), "up": up}
    if (alpha := present[0].get("alpha")) is not None:
        # Rank tripled, so alpha must too, or alpha/rank quietly divides the adapter by three.
        tensors["alpha"] = alpha * len(_QKV_PARTS)
    return tensors


def _split_qkv(
    block: str, tensors: dict[str, Any], layout: RowLayout
) -> dict[str, dict[str, Any]]:
    """The fused adapter back to three, each keeping the shared ``lora_A``."""
    up = tensors["up"]
    parts = len(_QKV_PARTS)
    if up.shape[0] % parts:
        raise ComponentError(f"{block}{_QKV} has {up.shape[0]} rows, not a multiple of {parts}.")
    if layout is RowLayout.INTERLEAVED:
        up = deinterleave_rows(up, parts, HEAD_DIM)
    rows = up.shape[0] // parts
    out: dict[str, dict[str, Any]] = {}
    for index, part in enumerate(_QKV_PARTS):
        split = {"down": tensors["down"], "up": up[index * rows : (index + 1) * rows]}
        if (alpha := tensors.get("alpha")) is not None:
            split["alpha"] = alpha
        out[f"{block}{part}"] = split
    return out


def _swap_halves(tensor: Any) -> Any:
    import torch

    half = tensor.shape[0] // 2
    if half * 2 != tensor.shape[0]:
        raise ComponentError("A gated FFN adapter must have an even number of output rows.")
    return torch.cat([tensor[half:], tensor[:half]], dim=0)
