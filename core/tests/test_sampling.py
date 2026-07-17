"""The model-family sampling registry: the torch-free param fields (always) and the diffusers-backed
``apply_sampling`` resolver (skipped when diffusers is absent, like the other runner tests)."""

from __future__ import annotations

from typing import Any

import pytest

from inline_core.graph.descriptor import Widget
from inline_core.models.sampling import (
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    SAMPLER_OPTIONS,
    SCHEDULER_OPTIONS,
    SamplingFamily,
    sampling_param_fields,
)

FAMILY = SamplingFamily.FLOW_MATCH


# --- torch-free data layer (no diffusers needed) -------------------------------------------------


def test_param_fields_are_advanced_selects_with_euler_simple_defaults() -> None:
    sampler, scheduler = sampling_param_fields(FAMILY)
    assert (sampler.key, scheduler.key) == ("sampler", "scheduler")
    assert sampler.widget is Widget.SELECT and scheduler.widget is Widget.SELECT
    assert sampler.default == DEFAULT_SAMPLER == "euler"
    assert scheduler.default == DEFAULT_SCHEDULER == "simple"
    assert sampler.advanced and scheduler.advanced
    assert sampler.options == SAMPLER_OPTIONS[FAMILY]
    assert scheduler.options == SCHEDULER_OPTIONS[FAMILY]
    # Defaults must be members of their own option sets (a saved node with no value gets them).
    assert DEFAULT_SAMPLER in {o.value for o in SAMPLER_OPTIONS[FAMILY]}
    assert DEFAULT_SCHEDULER in {o.value for o in SCHEDULER_OPTIONS[FAMILY]}


# --- resolver (diffusers required) ---------------------------------------------------------------

pytest.importorskip("diffusers")

from inline_core.models.sampling import apply_sampling  # noqa: E402


class _Pipe:
    """A stand-in pipeline: ``apply_sampling`` only reads/replaces ``.scheduler``."""

    def __init__(self, scheduler: Any = None) -> None:
        self.scheduler = scheduler


def _base_config() -> dict[str, Any]:
    from diffusers import FlowMatchEulerDiscreteScheduler

    return dict(FlowMatchEulerDiscreteScheduler().config)


@pytest.mark.parametrize(
    ("sampler", "scheduler", "flag", "value"),
    [
        ("euler", "simple", "stochastic_sampling", False),
        ("euler_a", "simple", "stochastic_sampling", True),
        ("euler", "karras", "use_karras_sigmas", True),
        ("euler", "exponential", "use_exponential_sigmas", True),
    ],
)
def test_apply_sampling_sets_expected_flag(
    sampler: str, scheduler: str, flag: str, value: bool
) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    base = _base_config()
    pipe = _Pipe()
    sigmas = apply_sampling(pipe, base, FAMILY, sampler, scheduler, steps=8)
    assert sigmas is None  # only `uniform` supplies sigmas
    assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler)
    assert pipe.scheduler.config[flag] == value


def test_beta_scheduler_sets_flag_when_scipy_present() -> None:
    # diffusers gates use_beta_sigmas on scipy at construction (declared in the `zimage` extra).
    pytest.importorskip("scipy")
    base = _base_config()
    pipe = _Pipe()
    assert apply_sampling(pipe, base, FAMILY, "euler", "beta", steps=8) is None
    assert pipe.scheduler.config["use_beta_sigmas"] is True


def test_flow_match_schedulers_supply_no_explicit_sigmas() -> None:
    # Every flow-match option is a config flag, so apply_sampling returns None (the `sigmas=` kwarg
    # is never added). `uniform` was dropped: Z-Image's default spacing already is a linear ramp.
    base = _base_config()
    for scheduler in ("simple", "karras", "exponential"):
        assert apply_sampling(_Pipe(), base, FAMILY, "euler", scheduler, steps=8) is None
    assert "uniform" not in {o.value for o in SCHEDULER_OPTIONS[FAMILY]}


def test_unknown_ids_fall_back_to_euler_simple() -> None:
    base = _base_config()
    pipe = _Pipe()
    # An old saved node with a value predating the curated set resolves to the defaults, no crash.
    sigmas = apply_sampling(pipe, base, FAMILY, "bogus", "nope", steps=8)
    assert sigmas is None
    assert pipe.scheduler.config["stochastic_sampling"] is False


def test_non_flow_match_base_is_left_untouched() -> None:
    # A whole-pipeline folder carrying some other scheduler class: the swap is skipped defensively.
    base = _base_config()
    base["_class_name"] = "DDIMScheduler"
    sentinel = object()
    pipe = _Pipe(sentinel)
    assert apply_sampling(pipe, base, FAMILY, "euler", "karras", steps=8) is None
    assert pipe.scheduler is sentinel  # unchanged


def test_unsupported_family_raises() -> None:
    with pytest.raises(ValueError, match="family"):
        apply_sampling(_Pipe(), _base_config(), "epsilon", "euler", "simple", 8)  # type: ignore[arg-type]
