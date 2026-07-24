"""Krea 2 runner: registration, per-variant defaults, and the model gate - pipeline mocked."""

from __future__ import annotations

import types
from typing import Any

import pytest

from inline_core.device.memory import MemoryPolicy
from inline_core.errors import ComponentError
from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import Node
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import CollectingEmitter

rk = pytest.importorskip("inline_core.models.krea2.runner")


class _FakeImage:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size


class _FakePipe:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> Any:
        self.calls.append(kw)
        callback = kw.get("callback_on_step_end")
        if callback:
            for i in range(kw["num_inference_steps"]):
                callback(self, i, None, {"latents": None})
        return types.SimpleNamespace(images=[_FakeImage((kw["width"], kw["height"]))])


class _FakeStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save(self, run_id: str, node_id: str, image: Any, params: dict[str, Any]) -> Any:
        self.saved.append({"run_id": run_id, "node_id": node_id, "params": params})
        from inline_core.media import MediaKind
        from inline_core.takes import Take

        return Take(
            id=f"take-{node_id}", run_id=run_id, node_id=node_id, kind=MediaKind.IMAGE,
            uri=f"mem://{node_id}", hash="h", params=dict(params),
        )


@pytest.fixture
def use_fake_pipe(monkeypatch: pytest.MonkeyPatch) -> _FakePipe:
    pipe = _FakePipe()
    monkeypatch.setattr(
        rk, "_load_pipeline",
        lambda policy, *, variant, source, vae, text, quant=None, loras=(), cancel_check=None: pipe,
    )
    monkeypatch.setattr(rk.reqs, "krea2_requirements", lambda variant, params=None: [])
    monkeypatch.setattr(rk.reqs, "resolve_diffusion", lambda variant, params=None: "krea2.st")
    monkeypatch.setattr(rk.reqs, "resolve_vae", lambda params=None: "vae.st")
    monkeypatch.setattr(rk.reqs, "resolve_text_encoder", lambda params=None: "te.st")
    return pipe


def _ctx() -> tuple[ExecutionContext, CollectingEmitter]:
    emitter = CollectingEmitter()
    ctx = ExecutionContext(
        run_id="run1", policy=MemoryPolicy(), emitter=emitter, cancel=CancelToken()
    )
    return ctx, emitter


def test_register_adds_both_nodes() -> None:
    registry = build_default_registry()
    rk.register_krea2(registry, _FakeStore(), MemoryPolicy())

    assert registry.has("krea/krea-2-turbo")
    assert registry.has("krea/krea-2-raw")
    descriptor = registry.get("krea/krea-2-turbo")
    assert [p.id for p in descriptor.inputs] == [
        "prompt", "model", "vae", "text_encoder", "lora", "image",
    ]
    assert descriptor.input("prompt").required
    assert descriptor.output_kind is not None


def test_turbo_is_distilled_and_raw_is_not() -> None:
    turbo, raw = rk.KREA2_TURBO.defaults(), rk.KREA2_RAW.defaults()

    # Turbo is the 8-step distilled checkpoint (CFG off); RAW needs the full schedule + guidance.
    assert (turbo["steps"], turbo["guidance"]) == (8, 0.0)
    assert (raw["steps"], raw["guidance"]) == (28, 4.5)


def test_generation_passes_the_variant_defaults_through(use_fake_pipe: _FakePipe) -> None:
    runner = rk.Krea2Runner(_FakeStore(), MemoryPolicy(), "raw")
    ctx, _ = _ctx()

    node = Node(id="n1", type="krea/krea-2-raw", params={"seed": 7})
    runner.run(node, {"prompt": ["a fox"]}, ctx)

    call = use_fake_pipe.calls[0]
    assert call["num_inference_steps"] == 28
    assert call["guidance_scale"] == 4.5
    assert (call["width"], call["height"]) == (1024, 1024)


def test_progress_is_emitted_per_step(use_fake_pipe: _FakePipe) -> None:
    runner = rk.Krea2Runner(_FakeStore(), MemoryPolicy(), "turbo")
    ctx, emitter = _ctx()

    runner.run(
        Node(id="n1", type="krea/krea-2-turbo", params={"steps": 4}), {"prompt": ["a fox"]}, ctx
    )

    steps = [e.step for e in emitter.events if e.step is not None]
    assert steps == [1, 2, 3, 4]


def test_the_take_records_what_produced_it(use_fake_pipe: _FakePipe) -> None:
    store = _FakeStore()
    runner = rk.Krea2Runner(store, MemoryPolicy(), "turbo")
    ctx, _ = _ctx()

    runner.run(
        Node(id="n1", type="krea/krea-2-turbo", params={"seed": 99}), {"prompt": ["a fox"]}, ctx
    )

    params = store.saved[0]["params"]
    assert params["seed"] == 99
    assert params["prompt"] == "a fox"
    assert params["model"] == "krea2.st"


def test_a_missing_prompt_is_a_clear_error(use_fake_pipe: _FakePipe) -> None:
    runner = rk.Krea2Runner(_FakeStore(), MemoryPolicy(), "turbo")
    ctx, _ = _ctx()

    with pytest.raises(ComponentError, match="needs a prompt"):
        runner.run(Node(id="n1", type="krea/krea-2-turbo", params={}), {}, ctx)


def test_missing_models_point_at_the_popup(monkeypatch: pytest.MonkeyPatch) -> None:
    component = types.SimpleNamespace(
        id="diffusion", label="Diffusion model (TURBO)", present=False
    )
    monkeypatch.setattr(rk.reqs, "krea2_requirements", lambda variant, params=None: [component])
    runner = rk.Krea2Runner(_FakeStore(), MemoryPolicy(), "turbo")
    ctx, _ = _ctx()

    with pytest.raises(ComponentError, match="model popup"):
        runner.run(Node(id="n1", type="krea/krea-2-turbo", params={}), {"prompt": ["a fox"]}, ctx)


def test_only_turbo_suggests_dropping_guidance_on_an_oom() -> None:
    turbo = rk.Krea2Runner(_FakeStore(), MemoryPolicy(), "turbo")
    raw = rk.Krea2Runner(_FakeStore(), MemoryPolicy(), "raw")

    # RAW genuinely needs CFG, so telling its user to set guidance 0 would wreck the output.
    assert "CFG-free" in turbo._oom(1024, 1024, guidance=4.5)
    assert "CFG-free" not in raw._oom(1024, 1024, guidance=4.5)
