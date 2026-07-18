"""Model-family-aware sampler & scheduler registry (reusable across Core model runners).

Two halves, split to honor the import-guard rule (see core/CLAUDE.md):

- A **torch-free data layer** - the curated sampler/scheduler options per model family plus the two
  ``SELECT`` param fields a runner splices into its descriptor. Importable anywhere; no torch/
  diffusers, only the plain ``graph.descriptor`` dataclasses.
- A **resolver** (``apply_sampling``) that rebuilds a pipeline's scheduler for a selected
  ``(sampler, scheduler)`` pair. diffusers is imported *lazily inside* the function.

Why a curated, family-specific set rather than the classic Stable-Diffusion sampler list: Z-Image is
a **flow-matching** model. Its diffusers pipeline ``__call__`` always forces ``sigmas=`` + ``mu=``
into ``scheduler.set_timesteps``, and only ``FlowMatchEulerDiscreteScheduler`` accepts both - so the
classic DPM++/UniPC/DDIM/Heun sampler *class-swap* is not physically valid on the stock pipeline.
For this family, selecting a **scheduler** flips a sigma-spacing config flag (Karras/Exponential/
Beta), and selecting a **sampler** flips the ancestral (``stochastic_sampling``) flag; everything
stays on diffusers' supported path. The family split lets epsilon models (SDXL) unlock the
class-swap set later without reworking the UI.

(A ``uniform`` scheduler was evaluated and dropped: Z-Image's default sigma spacing already IS a
linear ramp, so an explicit uniform ramp produced byte-identical output to ``simple`` at every step
count - a confusing no-op option. ``apply_sampling`` keeps its ``list[float] | None`` return as the
seam for a future family whose scheduler genuinely supplies explicit sigmas.)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from ..graph.descriptor import Option, ParamField, Widget

logger = logging.getLogger("inline_core.sampling")


class SamplingFamily(str, Enum):
    """The math family a model's scheduler belongs to. Only flow-match ships today; epsilon
    (SDXL-style class-swap samplers) is the intended next entry - add it here plus its option lists
    and flag maps, and every flow-match runner keeps working untouched."""

    FLOW_MATCH = "flow_match"


# --- curated options (ids align with the graph/primitives._SAMPLERS/_SCHEDULERS vocabulary) -------

_FLOW_MATCH_SAMPLERS: tuple[Option, ...] = (
    Option("euler", "Euler"),
    Option("euler_a", "Euler Ancestral"),
)

_FLOW_MATCH_SCHEDULERS: tuple[Option, ...] = (
    Option("simple", "Simple"),
    Option("karras", "Karras"),
    Option("exponential", "Exponential"),
    Option("beta", "Beta"),
)

SAMPLER_OPTIONS: dict[SamplingFamily, tuple[Option, ...]] = {
    SamplingFamily.FLOW_MATCH: _FLOW_MATCH_SAMPLERS,
}

SCHEDULER_OPTIONS: dict[SamplingFamily, tuple[Option, ...]] = {
    SamplingFamily.FLOW_MATCH: _FLOW_MATCH_SCHEDULERS,
}

DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# id -> the FlowMatchEulerDiscreteScheduler config flags it flips. Sampler and scheduler flags are
# merged; the sigma flags below are mutually exclusive by construction (each scheduler sets at most
# one), so the scheduler's ">1 sigma flag" guard can never trip.
_SAMPLER_FLAGS: dict[str, dict[str, Any]] = {
    "euler": {"stochastic_sampling": False},
    "euler_a": {"stochastic_sampling": True},  # ancestral
}

_SCHEDULER_FLAGS: dict[str, dict[str, Any]] = {
    "simple": {},  # model default spacing
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},  # needs scipy (declared in the `zimage` extra)
}


def sampling_param_fields(family: SamplingFamily) -> tuple[ParamField, ParamField]:
    """The two ``SELECT`` param fields (sampler, scheduler) a runner splices into its descriptor
    ``params``. Torch-free - plain dataclasses. Marked ``advanced`` so they live behind the node's
    Adjust panel and generation stays one-click at the default euler/simple."""
    return (
        ParamField(
            "sampler", "Sampler", Widget.SELECT, DEFAULT_SAMPLER,
            options=SAMPLER_OPTIONS[family], advanced=True,
        ),
        ParamField(
            "scheduler", "Scheduler", Widget.SELECT, DEFAULT_SCHEDULER,
            options=SCHEDULER_OPTIONS[family], advanced=True,
        ),
    )


def _validated(value: Any, options: tuple[Option, ...], default: str) -> str:
    """Coerce a saved id to a known option, else the family default - safety for an old saved node
    whose stored value predates a curated set (the descriptor ``defaults()`` merge already guards
    the common case; this guards a stale explicit value)."""
    ids = {o.value for o in options}
    text = str(value)
    return text if text in ids else default


def apply_sampling(
    pipe: Any,
    base_scheduler_config: Any,
    family: SamplingFamily,
    sampler_id: str,
    scheduler_id: str,
    steps: int,
) -> list[float] | None:
    """Rebuild ``pipe.scheduler`` for the selected ``(sampler, scheduler)`` pair.

    Returns an explicit ``sigmas`` list when a scheduler supplies its own spacing (none do for
    the flow-match family today - every option is a config flag - but the return type and ``steps``
    are the seam for a future family whose scheduler builds sigmas), else ``None``. The caller adds
    it to the pipeline call as ``sigmas=`` when present.

    ``base_scheduler_config`` is the pipeline's **original** scheduler config, captured once at load
    before any mutation. Rebuilding from it every run (a) preserves the model's trained
    ``shift``/``use_dynamic_shifting`` and (b) stops a prior run's sampler/scheduler leaking through
    the shared pipeline cache (the cache key does not include sampler/scheduler, and mutating
    ``pipe.scheduler`` persists across cache hits).

    Flow-match only: the sole compatible scheduler class is ``FlowMatchEulerDiscreteScheduler``.
    If the base config is another class (an unusual whole-pipeline folder), the swap is skipped
    and the pipeline keeps its own scheduler - defensive, unlikely for Z-Image. Any build failure
    (e.g. a missing optional dep for ``beta``) falls back to the model default spacing and logs,
    never breaking a generation."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    if family is not SamplingFamily.FLOW_MATCH:
        raise ValueError(f"Unsupported sampling family: {family!r}")

    sampler_id = _validated(sampler_id, SAMPLER_OPTIONS[family], DEFAULT_SAMPLER)
    scheduler_id = _validated(scheduler_id, SCHEDULER_OPTIONS[family], DEFAULT_SCHEDULER)

    base_cls = (
        base_scheduler_config.get("_class_name")
        if hasattr(base_scheduler_config, "get")
        else None
    )
    if base_cls is not None and base_cls != "FlowMatchEulerDiscreteScheduler":
        logger.warning(
            "Base scheduler %s is not flow-match; leaving the pipeline's scheduler unchanged.",
            base_cls,
        )
        return None

    flags = {**_SAMPLER_FLAGS[sampler_id], **_SCHEDULER_FLAGS[scheduler_id]}
    try:
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler_config, **flags)
    except Exception as error:  # noqa: BLE001 - a sampling choice must never break a generation
        logger.warning(
            "Could not build scheduler for sampler=%s scheduler=%s (%s); "
            "falling back to the model default spacing.",
            sampler_id,
            scheduler_id,
            error,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler_config)
        return None

    # No flow-match scheduler supplies explicit sigmas (see the module note on the dropped
    # `uniform`); `steps` is unused here but kept for the future-family seam described above.
    _ = steps
    return None
