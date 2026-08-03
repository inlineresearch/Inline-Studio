"""Load a checkpoint written for one implementation into a model built for another.

A key plan is *declared* - rename this, split that fused tensor into thirds, swap those halves, drop
that one - and applied while weights stream in, so peak memory stays near a single tensor and there
is no converted second copy on disk.

The reason this is a module and not a dict comprehension inside one runner: the transforms it
performs are the ones that fail **silently**. A fused QKV split the wrong way round, or a gated FFN
whose halves are swapped backwards, satisfies every shape check, fills every parameter, and renders
a video that plays. So the plan carries an expected row layout and ``detect_row_layout`` measures
the real one from the tensor's own statistics, turning that class of bug into a load error.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..errors import ComponentError


class RowLayout(str, Enum):
    """How the parts of a fused tensor are arranged down its first dimension."""

    #: ``[part0_all; part1_all; part2_all]`` - each part's rows contiguous.
    CONTIGUOUS = "contiguous"
    #: ``[p0_h0; p1_h0; p2_h0][p0_h1; ...]`` - parts interleaved per head.
    INTERLEAVED = "interleaved"


# --- the plan ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rename:
    """One source tensor becomes one target tensor, unchanged."""

    target: str


@dataclass(frozen=True)
class Split:
    """A fused tensor becomes several targets, taken as equal parts along dim 0.

    ``layout`` states how the source is arranged. A source that is interleaved per head is
    de-interleaved first; a contiguous one is sliced directly. Getting this backwards is the bug
    this module exists to catch, so it is never inferred silently - see ``detect_row_layout``.
    """

    targets: tuple[str, ...]
    layout: RowLayout = RowLayout.CONTIGUOUS
    #: Only meaningful when interleaved: how many rows each part contributes per head.
    head_dim: int = 0


@dataclass(frozen=True)
class SwapHalves:
    """A fused pair becomes one target with its two halves exchanged along dim 0.

    diffusers' ``SwiGLU`` reads ``[value; gate]`` where the reference checkpoints store
    ``[gate; value]``.
    """

    target: str


@dataclass(frozen=True)
class SplitWeightNorm:
    """A fused weight becomes the ``weight_g`` / ``weight_v`` pair ``nn.utils.weight_norm`` wants.

    Weight-normalised modules (BigVGAN-style audio decoders, most vocoders) store a magnitude and a
    direction rather than one weight, but publishers usually ship the fused product. The split is
    exact: ``v = W`` and ``g = ||W||`` over every dim but the first gives back ``g * v/||v||``.
    """

    magnitude: str
    direction: str


@dataclass(frozen=True)
class Drop:
    """A source tensor the target model does not want. ``reason`` is required, because an
    unexplained drop is indistinguishable from a plan that forgot a key."""

    reason: str


@dataclass(frozen=True)
class AssertEqual:
    """Keep the tensor out of the model but check it against what the model computed itself.

    ``rope.inv_freq`` is the case in point: it is a pure function of theta, ports recompute it, and
    the config does not state theta. Asserting the shipped values match the recomputed ones turns a
    wrong default into a load error rather than geometry that drifts across a frame.
    """

    attribute: str
    tolerance: float = 0.0


Action = Rename | Split | SwapHalves | SplitWeightNorm | Drop | AssertEqual


@dataclass(frozen=True)
class KeyPlan:
    """A complete source-to-target mapping, plus the version that invalidates prepared artifacts."""

    version: str
    actions: dict[str, Action] = field(default_factory=dict)

    def targets(self) -> set[str]:
        out: set[str] = set()
        for action in self.actions.values():
            if isinstance(action, Rename | SwapHalves):
                out.add(action.target)
            elif isinstance(action, Split):
                out.update(action.targets)
            elif isinstance(action, SplitWeightNorm):
                out.update((action.magnitude, action.direction))
        return out


def check_coverage(plan: KeyPlan, source_keys: Sequence[str], target_keys: Sequence[str]) -> None:
    """Every source key is accounted for and every target parameter is filled.

    Runs against a real checkpoint header, which costs a few hundred KB rather than a download, so a
    plan is proved complete before a runner is ever built on top of it.
    """
    unmapped = sorted(set(source_keys) - set(plan.actions))
    if unmapped:
        raise ComponentError(
            f"{len(unmapped)} checkpoint tensors have no action in key plan {plan.version}: "
            + ", ".join(unmapped[:5])
            + ("…" if len(unmapped) > 5 else "")
        )
    stale = sorted(set(plan.actions) - set(source_keys))
    if stale:
        raise ComponentError(
            f"Key plan {plan.version} maps {len(stale)} tensors this checkpoint does not have: "
            + ", ".join(stale[:5])
        )
    missing = sorted(set(target_keys) - plan.targets())
    if missing:
        raise ComponentError(
            f"{len(missing)} model parameters are left unfilled by key plan {plan.version}: "
            + ", ".join(missing[:5])
            + ("…" if len(missing) > 5 else "")
        )


# --- layout detection ----------------------------------------------------------------------------


def detect_row_layout(row_stat: Sequence[float], parts: int, head_dim: int) -> RowLayout:
    """Which arrangement the rows of a fused tensor are actually in.

    ``row_stat`` is one number per row: an L2 norm, an RMS, or the per-row quantisation scale,
    whichever is cheap for the caller. Q, K and V (or gate and up) have genuinely different weight
    magnitudes, so the true grouping is the one whose groups separate: between-group spread large
    against within-group spread. The false grouping mixes all the parts into every group and comes
    out flat.
    """
    rows = len(row_stat)
    if parts < 2 or head_dim < 1 or rows % (parts * head_dim):
        raise ValueError(f"{rows} rows do not divide into {parts} parts of {head_dim}.")
    # With a single head the two groupings are the same partition, so a verdict would be a coin
    # flip dressed up as a measurement. This is why the check needs the whole tensor, not a window
    # off the front of it: one head's worth of rows cannot tell the layouts apart.
    if rows // (parts * head_dim) < 2:
        raise ValueError(
            f"{rows} rows is one head; the layouts are indistinguishable below two heads."
        )
    contiguous = _separation(row_stat, _contiguous_groups(rows, parts))
    interleaved = _separation(row_stat, _interleaved_groups(rows, parts, head_dim))
    return RowLayout.CONTIGUOUS if contiguous >= interleaved else RowLayout.INTERLEAVED


def _contiguous_groups(rows: int, parts: int) -> list[int]:
    block = rows // parts
    return [index // block for index in range(rows)]


def _interleaved_groups(rows: int, parts: int, head_dim: int) -> list[int]:
    return [(index // head_dim) % parts for index in range(rows)]


def _separation(values: Sequence[float], groups: Sequence[int]) -> float:
    """Between-group variance over within-group variance, the usual one-way F statistic."""
    count = len(values)
    grand = sum(values) / count
    buckets: dict[int, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        buckets.setdefault(group, []).append(value)
    between = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in buckets.values())
    within = sum((x - sum(v) / len(v)) ** 2 for v in buckets.values() for x in v)
    if within <= 0:
        return math.inf if between > 0 else 0.0
    return between / within


def row_stats(tensor: Any) -> list[float]:
    """Per-row RMS of a 2D tensor, as the input to ``detect_row_layout``."""
    import numpy as np

    if hasattr(tensor, "detach"):  # a torch tensor
        import torch

        # numpy has no bfloat16, which is the dtype every one of these checkpoints ships in, so
        # this cast is required rather than defensive.
        source = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        source = tensor
    array = np.asarray(source)
    if array.ndim != 2:
        raise ValueError(f"Row statistics need a 2D tensor, got shape {array.shape}.")
    return [float(v) for v in np.sqrt((array.astype(np.float32) ** 2).mean(axis=1))]


def assert_layout(tensor: Any, split: Split, *, key: str) -> None:
    """Fail the load when a fused tensor is not arranged the way the plan says it is."""
    if split.head_dim < 1:
        return  # nothing to measure against; the plan did not claim a head layout
    actual = detect_row_layout(row_stats(tensor), len(split.targets), split.head_dim)
    if actual is not split.layout:
        raise ComponentError(
            f"{key} looks {actual.value} but the plan expects {split.layout.value}. Splitting it "
            "the wrong way renders a video that plays and is wrong, so this is refused. The "
            "checkpoint is probably from a different repack than the plan was written for."
        )


# --- applying the plan ---------------------------------------------------------------------------


def transform(
    key: str, tensor: Any, action: Action, *, verify_layout: bool = True
) -> Iterator[tuple[str, Any]]:
    """Yield the ``(target_key, tensor)`` pairs one source tensor becomes."""
    if isinstance(action, Drop | AssertEqual):
        return
    if isinstance(action, Rename):
        yield action.target, tensor
        return
    if isinstance(action, SplitWeightNorm):
        # g = ||W|| over every dim but the first, keeping the dims so it broadcasts back.
        dims = tuple(range(1, tensor.ndim))
        magnitude = tensor.float().pow(2).sum(dim=dims, keepdim=True).sqrt().to(tensor.dtype)
        yield action.magnitude, magnitude
        yield action.direction, tensor
        return
    if isinstance(action, SwapHalves):
        half = tensor.shape[0] // 2
        if half * 2 != tensor.shape[0]:
            raise ComponentError(f"{key} has an odd first dimension and cannot be halved.")
        yield action.target, _cat([tensor[half:], tensor[:half]])
        return
    parts = len(action.targets)
    if tensor.shape[0] % parts:
        raise ComponentError(f"{key} has {tensor.shape[0]} rows, not divisible into {parts} parts.")
    if verify_layout:
        assert_layout(tensor, action, key=key)
    source = _deinterleave(tensor, parts, action.head_dim) if (
        action.layout is RowLayout.INTERLEAVED
    ) else tensor
    block = source.shape[0] // parts
    for index, target in enumerate(action.targets):
        yield target, source[index * block : (index + 1) * block]


def _deinterleave(tensor: Any, parts: int, head_dim: int) -> Any:
    """``[p0_h0; p1_h0; p2_h0][p0_h1; …]`` to ``[p0_all; p1_all; p2_all]``.

    ``transpose`` is not the same call in torch and numpy - torch swaps two axes, numpy wants a full
    permutation - so the swap is spelled per backend rather than duck-typed.
    """
    if head_dim < 1:
        raise ComponentError("De-interleaving needs the head dimension the parts are grouped by.")
    heads = tensor.shape[0] // (parts * head_dim)
    reshaped = tensor.reshape(heads, parts, head_dim, *tensor.shape[1:])
    if _is_torch(tensor):
        moved = reshaped.transpose(0, 1).contiguous()
    else:
        import numpy as np

        moved = np.ascontiguousarray(np.swapaxes(reshaped, 0, 1))
    return moved.reshape(tensor.shape)


def _is_torch(value: Any) -> bool:
    """numpy 2 gave ndarray a ``.device``, so that is no longer the discriminator; ``detach`` is."""
    return hasattr(value, "detach")


def _cat(parts: list[Any]) -> Any:
    if _is_torch(parts[0]):
        import torch

        return torch.cat(parts, dim=0)
    import numpy as np

    return np.concatenate(parts, axis=0)
