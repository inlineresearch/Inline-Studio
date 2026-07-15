"""Z-Image runner: registration, input/param resolution, progress, and cancel — with the heavy
diffusers pipeline mocked so nothing is downloaded and no GPU is needed."""

from __future__ import annotations

import types
from typing import Any

import pytest

from inline_core.device.memory import MemoryPolicy
from inline_core.errors import CancelledError, ComponentError
from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import Node
from inline_core.models.zimage import runner as rz
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import CollectingEmitter, Phase, ProgressEvent


class _FakeImage:
    """Stands in for a PIL image; the fake store never touches disk."""

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
        rz, "_load_pipeline", lambda policy, *, img2img, source, single_file: pipe
    )
    return pipe


def _ctx(cancel: CancelToken | None = None) -> tuple[ExecutionContext, CollectingEmitter]:
    emitter = CollectingEmitter()
    ctx = ExecutionContext(
        run_id="run1", policy=MemoryPolicy(), emitter=emitter, cancel=cancel or CancelToken()
    )
    return ctx, emitter


def test_register_adds_descriptor_and_runner() -> None:
    registry = build_default_registry()
    rz.register_zimage(registry, _FakeStore(), MemoryPolicy())

    assert registry.has("alibaba/z-image-turbo")
    descriptor = registry.get("alibaba/z-image-turbo")
    assert descriptor.output_kind is not None
    assert [p.id for p in descriptor.inputs] == ["prompt", "image"]
    assert descriptor.input("prompt").required and not descriptor.input("image").required
    assert registry.runner("alibaba/z-image-turbo").produces_takes


def test_run_resolves_inputs_and_saves_take(use_fake_pipe: _FakePipe) -> None:
    store = _FakeStore()
    runner = rz.ZImageRunner(store, MemoryPolicy())
    ctx, emitter = _ctx()
    node = Node(
        id="frame1", type="alibaba/z-image-turbo",
        params={"steps": 4, "seed": 123, "width": 512, "height": 768},
    )

    result = runner.run(node, {"prompt": ["a neon city"]}, ctx)

    call = use_fake_pipe.calls[0]
    assert call["prompt"] == "a neon city"
    assert (call["width"], call["height"]) == (512, 768)
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 0.0  # turbo default: CFG off
    assert "negative_prompt" not in call  # empty negative is omitted
    assert result.takes[0] is result.outputs["image"]
    assert store.saved[0]["params"]["seed"] == 123
    ticks = [e for e in emitter.events if isinstance(e, ProgressEvent) and e.phase is Phase.SAMPLE]
    assert len(ticks) == 4 and ticks[-1].fraction == pytest.approx(1.0)


def test_run_without_prompt_fails(use_fake_pipe: _FakePipe) -> None:
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx()
    node = Node(id="f", type="alibaba/z-image-turbo")
    with pytest.raises(ComponentError):
        runner.run(node, {"prompt": [""]}, ctx)


def test_negative_prompt_passed_when_set(use_fake_pipe: _FakePipe) -> None:
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx()
    node = Node(id="f", type="alibaba/z-image-turbo", params={"negative_prompt": "blurry"})
    runner.run(node, {"prompt": ["cat"]}, ctx)
    assert use_fake_pipe.calls[0]["negative_prompt"] == "blurry"


def test_cancel_during_sampling_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cancel = CancelToken()

    class _CancellingPipe:
        def __call__(self, **kw: Any) -> Any:
            cancel.cancel()
            kw["callback_on_step_end"](self, 0, None, {"latents": None})
            raise AssertionError("callback should have raised before returning")

    monkeypatch.setattr(
        rz, "_load_pipeline",
        lambda policy, *, img2img, source, single_file: _CancellingPipe(),
    )
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx(cancel)
    node = Node(id="f", type="alibaba/z-image-turbo")
    with pytest.raises(CancelledError):
        runner.run(node, {"prompt": ["cat"]}, ctx)


def test_resolve_seed() -> None:
    assert rz._resolve_seed(42) == 42
    assert rz._resolve_seed(0) == 0
    assert 0 <= rz._resolve_seed(-1) <= rz._SEED_MAX  # random, non-negative
    assert 0 <= rz._resolve_seed("not-a-number") <= rz._SEED_MAX


def test_resolve_model_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))  # empty models root -> no local files
    monkeypatch.setenv("INLINE_ZIMAGE_MODEL", "some-org/Custom-Model")
    assert rz._resolve_model() == ("some-org/Custom-Model", False)  # repo id, not a single file
    monkeypatch.delenv("INLINE_ZIMAGE_MODEL")
    assert rz._resolve_model() == (rz._DEFAULT_MODEL, False)  # default when nothing local
