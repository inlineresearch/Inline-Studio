"""The decomposed Z-Image primitive runners (encode/text, latent/empty, sample, vae/decode,
vae/encode). Import-guarded: skipped without the ``zimage`` extra. The heavy diffusers loads are
stubbed (no weights on disk in CI), but the flow-match scheduler is real torch - so the sample loop,
its progress ticks, cancellation, and the tensor plumbing are exercised on CPU without a GPU.

Real weights are still needed for an end-to-end image; that is a GPU smoke test, not a unit test."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")

import torch  # noqa: E402

from inline_core.device.memory import MemoryPolicy  # noqa: E402
from inline_core.device.policy import Profile  # noqa: E402
from inline_core.device.types import Device, DeviceKind  # noqa: E402
from inline_core.errors import CancelledError, ComponentError, UnknownNodeType  # noqa: E402
from inline_core.graph.loader_runners import ComponentRef  # noqa: E402
from inline_core.graph.registry import build_default_registry  # noqa: E402
from inline_core.graph.schema import Node  # noqa: E402
from inline_core.media import MediaKind  # noqa: E402
from inline_core.models.zimage import primitives as zp  # noqa: E402
from inline_core.runtime.context import CancelToken, ExecutionContext  # noqa: E402
from inline_core.runtime.progress import CollectingEmitter, Phase, ProgressEvent  # noqa: E402
from inline_core.runtime.store import TakeStore  # noqa: E402
from inline_core.takes import AssetRef, Take  # noqa: E402

_PRIMITIVES = ("encode/text", "latent/empty", "sample", "vae/decode", "vae/encode")


class _FakeStore(TakeStore):
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save(self, run_id: str, node_id: str, image: Any, params: dict[str, Any]) -> Take:
        self.saved.append({"run_id": run_id, "node_id": node_id, "image": image, "params": params})
        return Take(
            id=f"take-{node_id}", run_id=run_id, node_id=node_id, kind=MediaKind.IMAGE,
            uri=f"mem://{node_id}", hash="h", params=dict(params),
        )


def _cpu_policy() -> MemoryPolicy:
    # Pin CPU/FP32 so the stubbed components stay on CPU even on a CUDA box - deterministic tests.
    return MemoryPolicy(Device(DeviceKind.CPU), profile=Profile.CPU)


def _ctx(cancel: CancelToken | None = None) -> tuple[ExecutionContext, CollectingEmitter]:
    emitter = CollectingEmitter()
    ctx = ExecutionContext(
        run_id="run1", policy=_cpu_policy(), emitter=emitter, cancel=cancel or CancelToken()
    )
    return ctx, emitter


# --- registration --------------------------------------------------------------------------------


def test_register_unhides_primitives_with_runners() -> None:
    registry = build_default_registry()
    zp.register_zimage_primitives(registry, _FakeStore(), _cpu_policy())
    for node_type in _PRIMITIVES:
        assert registry.get(node_type).hidden is False, node_type
        assert registry.runner(node_type) is not None, node_type
    # Only vae/decode produces media takes; the other primitives emit opaque engine values.
    assert registry.runner("vae/decode").produces_takes is True
    for node_type in ("encode/text", "latent/empty", "sample", "vae/encode"):
        assert registry.runner(node_type).produces_takes is False


def test_primitives_stay_hidden_without_the_runner() -> None:
    # Without register_zimage_primitives (the torch-less boot), the five stay hidden descriptors.
    registry = build_default_registry()
    for node_type in _PRIMITIVES:
        assert registry.get(node_type).hidden is True, node_type
    with pytest.raises(UnknownNodeType):
        registry.runner("sample")


# --- latent/empty (real, CPU) --------------------------------------------------------------------


def test_empty_latent_produces_expected_shape() -> None:
    runner = zp.EmptyLatentRunner(_cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="lat", type="latent/empty", params={"width": 1024, "height": 512, "batch": 2})
    latents = runner.run(node, {}, ctx).outputs["latent"]
    # 16 latent channels; spatial = 2 * (px // 16): 1024 -> 128, 512 -> 64.
    assert latents.tensor.shape == (2, 16, 64, 128)
    assert latents.tensor.dtype == torch.float32
    assert bool(torch.count_nonzero(latents.tensor) == 0)  # zeros; sample adds the seeded noise


# --- encode/text ---------------------------------------------------------------------------------


class _FakeTokenizer:
    def apply_chat_template(self, messages: Any, **kw: Any) -> str:
        return messages[0]["content"]

    def __call__(self, prompts: list[str], **kw: Any) -> Any:
        # Two tokens; the second is padding (mask 0) so the mask-slicing keeps only real tokens.
        import types

        return types.SimpleNamespace(
            input_ids=torch.tensor([[1, 0]]),
            attention_mask=torch.tensor([[1, 0]]),
        )


class _FakeTextEncoder:
    def to(self, device: str) -> _FakeTextEncoder:
        return self

    def __call__(self, *, input_ids: Any, attention_mask: Any, output_hidden_states: bool) -> Any:
        import types

        # Two hidden layers so hidden_states[-2] is well-defined; batch 1, seq 2, hidden 4.
        layer = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        return types.SimpleNamespace(hidden_states=[layer, layer])


def test_encode_text_requires_a_prompt() -> None:
    runner = zp.EncodeTextRunner(_cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="e", type="encode/text")
    te = ComponentRef(kind="text_encoder", arch="z-image", file="/m/te.safetensors")
    with pytest.raises(ComponentError, match="prompt"):
        runner.run(node, {"text_encoder": [te], "prompt": [""]}, ctx)


def test_encode_text_rejects_miswired_handle() -> None:
    runner = zp.EncodeTextRunner(_cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="e", type="encode/text")
    vae = ComponentRef(kind="vae", arch="z-image", file="/m/ae.safetensors")
    with pytest.raises(ComponentError, match="text_encoder handle"):
        runner.run(node, {"text_encoder": [vae], "prompt": ["a fox"]}, ctx)


def _load_te_stub(*args: Any, **kw: Any) -> tuple[Any, Any]:
    return _FakeTextEncoder(), _FakeTokenizer()


def test_encode_text_masks_and_produces_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zp.loaders, "load_text_encoder", _load_te_stub)
    runner = zp.EncodeTextRunner(_cpu_policy())
    ctx, emitter = _ctx()
    node = Node(id="e", type="encode/text")
    te = ComponentRef(kind="text_encoder", arch="z-image", file="/m/te.safetensors")
    result = runner.run(node, {"text_encoder": [te], "prompt": ["a fox"]}, ctx)
    cond = result.outputs["conditioning"]
    assert isinstance(cond, zp.ZImageConditioning)
    # One prompt -> one embedding tensor; the padded (mask 0) token is dropped, so seq len 1.
    assert len(cond.positive) == 1
    assert cond.positive[0].shape == (1, 4)
    assert any(isinstance(e, ProgressEvent) and e.phase is Phase.ENCODE for e in emitter.events)


# --- sample --------------------------------------------------------------------------------------


class _FakeTransformer:
    """Stands in for ZImageTransformer2DModel: echoes the packed latents as the velocity, so the
    real flow-match scheduler drives a genuine (if trivial) denoise on CPU."""

    dtype = torch.float32

    def to(self, device: str) -> _FakeTransformer:
        return self

    def __call__(self, x_list: list[torch.Tensor], t: Any, embeds: Any, return_dict: bool) -> Any:
        return (list(x_list),)


def _load_diffusion_stub(*args: Any, **kw: Any) -> Any:
    return _FakeTransformer()


def _load_scheduler_stub(*args: Any, **kw: Any) -> Any:
    # Real scheduler, no assets/weights needed - the flow-match maths is genuinely exercised.
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    return FlowMatchEulerDiscreteScheduler()


def _patch_sample_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zp.loaders, "load_diffusion", _load_diffusion_stub)
    monkeypatch.setattr(zp.loaders, "load_scheduler", _load_scheduler_stub)


def _sample_inputs() -> dict[str, list[Any]]:
    return {
        "model": [ComponentRef(kind="diffusion", arch="z-image", file="/m/diff.safetensors")],
        "positive": [zp.ZImageConditioning([torch.randn(3, 4)])],
        "latent": [zp.Latents(torch.zeros(1, 16, 8, 8, dtype=torch.float32))],
    }


def test_sample_runs_loop_and_streams_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sample_loads(monkeypatch)
    runner = zp.SampleRunner(_cpu_policy())
    ctx, emitter = _ctx()
    node = Node(id="s", type="sample", params={"steps": 3, "cfg": 0.0, "seed": 7})
    result = runner.run(node, _sample_inputs(), ctx).outputs["latent"]
    assert isinstance(result, zp.Latents)
    assert result.tensor.shape == (1, 16, 8, 8)
    ticks = [e for e in emitter.events if isinstance(e, ProgressEvent) and e.phase is Phase.SAMPLE]
    assert len(ticks) == 3 and ticks[-1].fraction == pytest.approx(1.0)


def test_sample_rejects_miswired_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sample_loads(monkeypatch)
    runner = zp.SampleRunner(_cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="s", type="sample")
    inputs = _sample_inputs()
    inputs["model"] = [ComponentRef(kind="vae", arch="z-image", file="/m/ae.safetensors")]
    with pytest.raises(ComponentError, match="diffusion handle"):
        runner.run(node, inputs, ctx)


def test_sample_requires_positive_and_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sample_loads(monkeypatch)
    runner = zp.SampleRunner(_cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="s", type="sample")
    inputs = _sample_inputs()
    del inputs["positive"]
    with pytest.raises(ComponentError, match="positive"):
        runner.run(node, inputs, ctx)
    inputs = _sample_inputs()
    del inputs["latent"]
    with pytest.raises(ComponentError, match="latent"):
        runner.run(node, inputs, ctx)


def test_sample_honors_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sample_loads(monkeypatch)
    cancel = CancelToken()
    cancel.cancel()
    runner = zp.SampleRunner(_cpu_policy())
    ctx, _ = _ctx(cancel)
    node = Node(id="s", type="sample", params={"steps": 3})
    with pytest.raises(CancelledError):
        runner.run(node, _sample_inputs(), ctx)


# --- vae/decode + vae/encode ---------------------------------------------------------------------


class _FakeConfig:
    scaling_factor = 0.3611
    shift_factor = 0.1159


class _FakeVAE:
    dtype = torch.float32
    config = _FakeConfig()

    def to(self, device: str) -> _FakeVAE:
        return self

    def decode(self, latents: torch.Tensor, return_dict: bool) -> Any:
        # A small, valid image tensor in [-1, 1] the real image processor can post-process to PIL.
        return (torch.zeros(latents.shape[0], 3, 64, 64),)

    def encode(self, image: torch.Tensor) -> Any:
        import types

        dist = types.SimpleNamespace(sample=lambda: torch.zeros(image.shape[0], 16, 8, 8))
        return types.SimpleNamespace(latent_dist=dist)


def _load_vae_stub(*args: Any, **kw: Any) -> Any:
    return _FakeVAE()


def test_vae_decode_saves_take(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zp.loaders, "load_vae", _load_vae_stub)
    store = _FakeStore()
    runner = zp.VaeDecodeRunner(store, _cpu_policy())
    ctx, emitter = _ctx()
    node = Node(id="d", type="vae/decode")
    inputs = {
        "vae": [ComponentRef(kind="vae", arch="z-image", file="/m/ae.safetensors")],
        "latent": [zp.Latents(torch.zeros(1, 16, 8, 8, dtype=torch.float32))],
    }
    result = runner.run(node, inputs, ctx)
    assert result.takes and result.takes[0] is result.outputs["image"]
    assert store.saved[0]["params"]["vae"].endswith("ae.safetensors")
    assert any(isinstance(e, ProgressEvent) and e.phase is Phase.DECODE for e in emitter.events)


def test_vae_decode_requires_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zp.loaders, "load_vae", _load_vae_stub)
    runner = zp.VaeDecodeRunner(_FakeStore(), _cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="d", type="vae/decode")
    vae = ComponentRef(kind="vae", arch="z-image", file="/m/ae.safetensors")
    with pytest.raises(ComponentError, match="latent"):
        runner.run(node, {"vae": [vae]}, ctx)


def test_vae_encode_requires_image_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zp.loaders, "load_vae", _load_vae_stub)
    runner = zp.VaeEncodeRunner(_cpu_policy())
    ctx, _ = _ctx()
    node = Node(id="ve", type="vae/encode")
    vae = ComponentRef(kind="vae", arch="z-image", file="/m/ae.safetensors")
    # An asset-id ref (no path) is not readable here - the runner asks for a path input.
    inputs = {"vae": [vae], "image": [AssetRef(ref="asset", id="x")]}
    with pytest.raises(ComponentError, match="image path"):
        runner.run(node, inputs, ctx)
