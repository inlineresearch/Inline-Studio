"""PipelineCache.evict_stale: switching what's resident (a different quant or a ControlNet) must
drop the previous pipeline so two multi-GB models never coexist, while an i2i build still reuses its
cached t2i base. Torch-free: the VRAM helpers are stubbed."""

from __future__ import annotations

import inline_core.models.pipeline_runtime as rt
from inline_core.models.pipeline_runtime import PipelineCache, PipelineKey


def _key(**over) -> PipelineKey:
    base = dict(
        arch="z-image",
        diffusion="/m.safetensors",
        vae="/ae.safetensors",
        text_encoder="/qwen.safetensors",
        variant="t2i",
        quant="none",
        loras=(),
        controlnet="",
    )
    base.update(over)
    return PipelineKey(**base)


def _stub(monkeypatch) -> list[dict]:
    """Record unload_components calls; neutralise free_vram (no torch)."""
    calls: list[dict] = []
    from inline_core.models import loaders

    monkeypatch.setattr(
        loaders,
        "unload_components",
        lambda **kw: calls.append(kw),
    )
    monkeypatch.setattr(rt, "free_vram", lambda: None)
    return calls


def test_control_run_evicts_the_plain_pipeline(monkeypatch) -> None:
    calls = _stub(monkeypatch)
    cache = PipelineCache()
    plain = _key()  # NONE quant, no controlnet
    cache.put(plain, object())

    control = _key(quant="int8", controlnet="/union.safetensors")
    cache.evict_stale(control)

    assert cache.get(plain) is None  # the resident full-precision pipeline is gone
    # And the component sweep is told to keep only the control build's files at its quant.
    assert calls and calls[0]["keep_quant"] == "int8"
    assert "/union.safetensors" in calls[0]["keep_files"]


def test_plain_run_evicts_the_control_pipeline(monkeypatch) -> None:
    _stub(monkeypatch)
    cache = PipelineCache()
    control = _key(quant="int8", controlnet="/union.safetensors")
    cache.put(control, object())

    cache.evict_stale(_key())  # back to a plain NONE-quant run
    assert cache.get(control) is None


def test_i2i_build_keeps_its_cached_t2i_base(monkeypatch) -> None:
    _stub(monkeypatch)
    cache = PipelineCache()
    t2i = _key(variant="t2i")
    cache.put(t2i, object())

    cache.evict_stale(_key(variant="i2i"))  # same weights/quant/controlnet, only shape differs
    assert cache.get(t2i) is not None  # reused, not evicted


def test_a_different_lora_stack_is_evicted(monkeypatch) -> None:
    _stub(monkeypatch)
    cache = PipelineCache()
    a = _key(loras=("lora-a@1.0",))
    cache.put(a, object())

    cache.evict_stale(_key(loras=("lora-b@1.0",)))
    assert cache.get(a) is None
