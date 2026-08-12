"""When the text encoder is evicted to the CPU for the denoise, and when it stays put.

Parking costs a GPU->CPU copy now plus a CPU->GPU copy on the next run. On a small card that buys
the denoise room it genuinely needs; on a large one it is seconds of PCIe traffic for headroom that
was never contended, which is what this pins.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from inline_core.device.policy import Profile  # noqa: E402
from inline_core.models.pipeline_runtime import module_bytes, should_park_encoder  # noqa: E402


class _Policy:
    def __init__(self, profile: Profile, free_mb: int | None) -> None:
        self._profile = profile
        self._free_mb = free_mb

    @property
    def profile(self) -> Profile:
        return self._profile

    def free_vram_mb(self) -> int | None:
        return self._free_mb


class _Encoder:
    """Stands in for a text encoder; only its size is read."""

    def __init__(self, n_bytes: int = 0) -> None:
        import torch

        self._p = [torch.zeros(max(1, n_bytes // 4), dtype=torch.float32)]

    def parameters(self):  # type: ignore[no-untyped-def]
        return iter(self._p)

    def buffers(self):  # type: ignore[no-untyped-def]
        return iter([])


def test_a_big_card_keeps_the_encoder_resident() -> None:
    # 80GB free: parking would trade seconds of copying for headroom nothing is competing for.
    policy = _Policy(Profile.GPU_MAX, 80 * 1024)
    assert should_park_encoder(policy, _Encoder()) is False


def test_a_tight_card_parks_the_encoder() -> None:
    policy = _Policy(Profile.GPU_MAX, 2 * 1024)
    assert should_park_encoder(policy, _Encoder()) is True


def test_the_lowvram_profile_always_parks() -> None:
    """The profile is an explicit user choice, so it wins over a momentarily healthy free figure."""
    policy = _Policy(Profile.LOWVRAM, 80 * 1024)
    assert should_park_encoder(policy, _Encoder()) is True


def test_an_unmeasurable_card_keeps_the_conservative_behaviour() -> None:
    policy = _Policy(Profile.GPU_MAX, None)
    assert should_park_encoder(policy, _Encoder()) is True


def test_module_bytes_counts_parameters() -> None:
    import torch

    linear = torch.nn.Linear(4, 4, bias=False)  # 16 fp32 params
    assert module_bytes(linear) == 16 * 4


def test_module_bytes_never_raises_on_an_odd_object() -> None:
    # It only feeds a heuristic, so a module that does not behave must not break a run.
    assert module_bytes(object()) == 0
