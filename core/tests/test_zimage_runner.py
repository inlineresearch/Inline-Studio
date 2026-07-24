"""Z-Image runner: registration, input/param resolution, progress, and cancel - with the heavy
diffusers pipeline mocked so nothing is downloaded and no GPU is needed."""

from __future__ import annotations

import types
from typing import Any

import pytest

from inline_core.device.memory import MemoryPolicy
from inline_core.device.types import Device, DeviceKind
from inline_core.errors import CancelledError, ComponentError
from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import Node
from inline_core.models import pipeline_runtime as rt
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
        rz,
        "_load_pipeline",
        lambda policy, *, img2img, source, mode, vae, text, quant=None, loras=(),
        cancel_check=None: pipe,
    )
    # These tests mock the pipeline, so bypass the "models present on disk" gate.
    monkeypatch.setattr(rz.reqs, "zimage_requirements", lambda params=None: [])
    monkeypatch.setattr(rz.reqs, "resolve_diffusion", lambda params=None: ("single_file", "fake"))
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
    assert [p.id for p in descriptor.inputs] == [
        "prompt", "model", "vae", "text_encoder", "lora", "image",
    ]
    assert descriptor.input("prompt").required
    # The component handles + lora + image are all optional (wire a Load node, or the dropdowns).
    for optional in ("model", "vae", "text_encoder", "lora", "image"):
        assert not descriptor.input(optional).required
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


def test_wired_component_handles_override_dropdowns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load/* node wired into model/vae/text_encoder feeds its file straight into the pipeline,
    bypassing the dropdown resolution."""
    from inline_core.graph.loader_runners import ComponentRef

    captured: dict[str, Any] = {}

    def _fake_load(
        policy: Any, *, img2img: bool, source: str, mode: str, vae: str, text: str,
        quant: Any = None, loras: Any = (), cancel_check: Any = None,
    ) -> Any:
        captured.update(source=source, mode=mode, vae=vae, text=text)
        return _FakePipe()

    monkeypatch.setattr(rz, "_load_pipeline", _fake_load)
    monkeypatch.setattr(rz.reqs, "zimage_requirements", lambda params=None: [])
    # The dropdown path must NOT be consulted when everything is wired.
    monkeypatch.setattr(
        rz.reqs, "resolve_diffusion", lambda params=None: pytest.fail("dropdown used despite wire")
    )
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx()
    node = Node(id="f", type="alibaba/z-image-turbo")
    inputs = {
        "prompt": ["a fox"],
        "model": [ComponentRef(kind="diffusion", arch="z-image", file="/m/diff.safetensors")],
        "vae": [ComponentRef(kind="vae", arch="z-image", file="/m/ae.safetensors")],
        "text_encoder": [
            ComponentRef(kind="text_encoder", arch="z-image", file="/m/qwen.safetensors")
        ],
    }
    runner.run(node, inputs, ctx)
    assert captured == {
        "source": "/m/diff.safetensors",
        "mode": "single_file",
        "vae": "/m/ae.safetensors",
        "text": "/m/qwen.safetensors",
    }


def test_miswired_handle_is_rejected(use_fake_pipe: _FakePipe) -> None:
    from inline_core.graph.loader_runners import ComponentRef

    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx()
    node = Node(id="f", type="alibaba/z-image-turbo")
    # A vae handle on the model port (the validator blocks this by kind; the runner guards too).
    inputs = {"prompt": ["x"], "model": [ComponentRef(kind="vae", arch="z-image", file="/m/ae")]}
    with pytest.raises(ComponentError, match="diffusion handle"):
        runner.run(node, inputs, ctx)


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
        lambda policy, *, img2img, source, mode, vae, text, quant=None, loras=(),
        cancel_check=None: (_CancellingPipe()),
    )
    monkeypatch.setattr(rz.reqs, "zimage_requirements", lambda params=None: [])
    monkeypatch.setattr(rz.reqs, "resolve_diffusion", lambda params=None: ("single_file", "fake"))
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx(cancel)
    node = Node(id="f", type="alibaba/z-image-turbo")
    with pytest.raises(CancelledError):
        runner.run(node, {"prompt": ["cat"]}, ctx)


def test_cancel_before_load_skips_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt that lands before the run starts must bail without loading a 12GB model."""
    def _must_not_load(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("_load_pipeline must not run once the run is already cancelled")

    monkeypatch.setattr(rz, "_load_pipeline", _must_not_load)
    monkeypatch.setattr(rz.reqs, "zimage_requirements", lambda params=None: [])
    monkeypatch.setattr(rz.reqs, "resolve_diffusion", lambda params=None: ("single_file", "fake"))
    cancel = CancelToken()
    cancel.cancel()  # already cancelled before we even start
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx(cancel)
    node = Node(id="f", type="alibaba/z-image-turbo")
    with pytest.raises(CancelledError):
        runner.run(node, {"prompt": ["cat"]}, ctx)


def test_cancel_during_load_raises_via_cancel_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner hands _load_pipeline a cancel_check that raises once the token is set, so an
    interrupt mid-load bails between components instead of only at the first denoise step."""
    cancel = CancelToken()

    def _fake_load(policy: Any, *, cancel_check: Any = None, **_k: Any) -> Any:
        cancel.cancel()  # user interrupts partway through the load
        cancel_check()  # the between-components checkpoint must now raise
        raise AssertionError("cancel_check should have raised")

    monkeypatch.setattr(rz, "_load_pipeline", _fake_load)
    monkeypatch.setattr(rz.reqs, "zimage_requirements", lambda params=None: [])
    monkeypatch.setattr(rz.reqs, "resolve_diffusion", lambda params=None: ("single_file", "fake"))
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx(cancel)
    node = Node(id="f", type="alibaba/z-image-turbo")
    with pytest.raises(CancelledError):
        runner.run(node, {"prompt": ["cat"]}, ctx)


def test_resolve_seed() -> None:
    assert rt.resolve_seed(42) == 42
    assert rt.resolve_seed(0) == 0
    assert 0 <= rt.resolve_seed(-1) <= rt._SEED_MAX  # random, non-negative
    assert 0 <= rt.resolve_seed("not-a-number") <= rt._SEED_MAX


def test_missing_models_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """An empty models dir must error clearly (pointing at the popup), never trigger a download."""
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))  # empty root -> everything missing
    monkeypatch.delenv("INLINE_ZIMAGE_MODEL", raising=False)
    runner = rz.ZImageRunner(_FakeStore(), MemoryPolicy())
    ctx, _ = _ctx()
    node = Node(id="f", type="alibaba/z-image-turbo")
    with pytest.raises(ComponentError, match="missing"):
        runner.run(node, {"prompt": ["cat"]}, ctx)


# --- text-encoder offload (reclaim its VRAM for the denoise) -------------------------------------

_CUDA = Device(DeviceKind.CUDA, 0)
_CPU = Device(DeviceKind.CPU)


class _FakeTextEncoder:
    """Records the device it was moved to, so a test can assert it was parked on the CPU."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: Any) -> _FakeTextEncoder:
        self.moves.append(str(device))
        return self


class _FakeEmbed:
    """A stand-in embedding tensor that records the device it was last moved to, so a test can
    assert the CPU-encoded embeddings were shipped back to the GPU for the denoise."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.device: str | None = None

    def to(self, device: Any) -> _FakeEmbed:
        self.device = str(device)
        return self


class _FakeEmbedPipe:
    """A pipe that supports precomputed embeddings (has ``prompt_embeds`` in ``__call__`` and an
    ``encode_prompt``), so ``_prompt_kwargs`` takes the encode-then-offload path."""

    def __init__(self, *, fail_encode: bool = False) -> None:
        self.text_encoder = _FakeTextEncoder()
        self.transformer = _FakeTextEncoder()  # records eviction/restore around the encode
        self.encode_calls: list[dict[str, Any]] = []
        self._fail = fail_encode

    def encode_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: Any = None,
        do_classifier_free_guidance: bool = True,
        device: Any = None,
    ) -> tuple[Any, Any]:
        if self._fail:
            raise RuntimeError("boom")
        self.encode_calls.append(
            {
                "prompt": prompt,
                "negative": negative_prompt,
                "cfg": do_classifier_free_guidance,
                "device": str(device),
            }
        )
        neg = [_FakeEmbed("neg-embed")] if do_classifier_free_guidance else []
        return [_FakeEmbed("pos-embed")], neg

    def __call__(  # noqa: D401 - signature is what _supports_prompt_embeds inspects
        self,
        prompt: Any = None,
        prompt_embeds: Any = None,
        negative_prompt: Any = None,
        negative_prompt_embeds: Any = None,
        **kw: Any,
    ) -> Any:  # pragma: no cover - not called in these unit tests
        return None


def test_prompt_kwargs_offloads_text_encoder_on_resident_gpu() -> None:
    """On a resident GPU placement the prompt is pre-encoded **on the GPU** (torchao int8 has a real
    CUDA matmul kernel, so the dequant is transient - a CPU encode instead OOM-kills a 16 GB host),
    then the encoder is parked on the CPU to free its VRAM for the denoise; the pipe is handed
    embeddings, not a raw prompt."""
    policy = MemoryPolicy(_CUDA, vram_gb=15.6)  # resident, no offload
    assert policy.placement("denoiser").offload is False
    pipe = _FakeEmbedPipe()

    kwargs = rz._prompt_kwargs(pipe, policy, prompt="a cat", negative=None, guidance=0.0)

    embeds = kwargs["prompt_embeds"]  # CFG off -> no negatives, no raw prompt
    assert kwargs.keys() == {"prompt_embeds"}
    assert [e.name for e in embeds] == ["pos-embed"]
    assert [e.device for e in embeds] == ["cuda:0"]  # on the GPU for the denoise
    assert pipe.encode_calls and pipe.encode_calls[0]["prompt"] == "a cat"
    assert pipe.encode_calls[0]["device"] == "cuda:0"  # encoded on the card (int8 CUDA kernel)
    # ensure-resident then park: on the card for the encode, off it for the denoise.
    assert pipe.text_encoder.moves == ["cuda:0", "cpu"]
    # the int8 transformer is left resident (torchao .to() round-trip is unreliable), not moved.
    assert pipe.transformer.moves == []


def test_prompt_kwargs_passes_negatives_when_cfg_on() -> None:
    """With guidance > 0 the pipeline needs the negative embeddings alongside the positives - both
    encoded on the GPU, then the encoder is parked on the CPU for the denoise."""
    policy = MemoryPolicy(_CUDA, vram_gb=15.6)
    pipe = _FakeEmbedPipe()

    kwargs = rz._prompt_kwargs(pipe, policy, prompt="a cat", negative="blurry", guidance=5.0)

    assert kwargs.keys() == {"prompt_embeds", "negative_prompt_embeds"}
    assert [e.name for e in kwargs["prompt_embeds"]] == ["pos-embed"]
    assert [e.name for e in kwargs["negative_prompt_embeds"]] == ["neg-embed"]
    assert [e.device for e in kwargs["negative_prompt_embeds"]] == ["cuda:0"]
    assert pipe.text_encoder.moves == ["cuda:0", "cpu"]


def test_prompt_kwargs_raw_prompt_when_offloaded() -> None:
    """An offload placement lets accelerate stream the encoder - pass the raw prompt, don't touch
    the text encoder ourselves."""
    policy = MemoryPolicy(_CUDA, vram_gb=6, allow_offload=True)  # MODEL offload
    assert policy.placement("denoiser").offload is True
    pipe = _FakeEmbedPipe()

    kwargs = rz._prompt_kwargs(pipe, policy, prompt="a cat", negative="blurry", guidance=0.0)

    assert kwargs == {"prompt": "a cat", "negative_prompt": "blurry"}
    assert pipe.encode_calls == []
    assert pipe.text_encoder.moves == []  # untouched


def test_prompt_kwargs_raw_prompt_on_cpu() -> None:
    policy = MemoryPolicy(_CPU, ram_gb=32)
    pipe = _FakeEmbedPipe()

    kwargs = rz._prompt_kwargs(pipe, policy, prompt="a cat", negative=None, guidance=0.0)

    assert kwargs == {"prompt": "a cat"}
    assert pipe.encode_calls == []


def test_prompt_kwargs_falls_back_when_encode_fails() -> None:
    """A failure in the offload optimization must degrade to the raw-prompt path, never raise."""
    policy = MemoryPolicy(_CUDA, vram_gb=15.6)
    pipe = _FakeEmbedPipe(fail_encode=True)

    kwargs = rz._prompt_kwargs(pipe, policy, prompt="a cat", negative="blurry", guidance=0.0)

    assert kwargs == {"prompt": "a cat", "negative_prompt": "blurry"}


def test_oom_message_flags_cfg_when_guidance_on() -> None:
    """The 1024² OOM on a T4 is CFG doubling the denoise batch - the message must say so and point
    at guidance=0 (turbo runs CFG-free), not just resolution."""
    with_cfg = rz._oom(1024, 1024, guidance=1.0)
    assert "Guidance" in with_cfg and "CFG-free" in with_cfg
    without_cfg = rz._oom(1024, 1024, guidance=0.0)
    assert "Guidance (CFG)" not in without_cfg  # no CFG hint when guidance is already off
    assert "Lower the resolution" in without_cfg
    # Host-RAM OOM is a load-time failure, unrelated to CFG.
    assert "Guidance" not in rz._oom(1024, 1024, host=True, guidance=1.0)


def test_text_encoder_detached_restores() -> None:
    """When active, the encoder is removed for the denoise (so device inference sees only the CUDA
    vae+transformer, not the CPU-parked encoder) and restored afterwards; inactive is a no-op."""
    pipe = _FakeEmbedPipe()
    original = pipe.text_encoder

    with rt.text_encoder_detached(pipe, True):
        assert pipe.text_encoder is None  # detached for the denoise call
    assert pipe.text_encoder is original  # restored for the next run's encode

    with rt.text_encoder_detached(pipe, False):
        assert pipe.text_encoder is original  # raw-prompt path: untouched
    assert pipe.text_encoder is original


def test_text_encoder_detached_restores_on_error() -> None:
    """A failure inside the denoise must still restore the encoder (finally), or the cached pipeline
    could never encode again."""
    pipe = _FakeEmbedPipe()
    original = pipe.text_encoder
    with pytest.raises(RuntimeError):
        with rt.text_encoder_detached(pipe, True):
            raise RuntimeError("denoise boom")
    assert pipe.text_encoder is original


class _FakeVae:
    """A stand-in AutoencoderKL carrying the tile thresholds ``_shrink_vae_tiles`` tunes."""

    def __init__(self) -> None:
        self.tile_sample_min_size = 1024
        self.tile_latent_min_size = 128  # sample_size / 8 - the Z-Image VAE's default


def test_shrink_vae_tiles_forces_tiling_at_1024() -> None:
    """The default tile threshold (128) equals the 1024² latent, so decode's strict ``>`` gate never
    tiles → full-frame OOM. Shrinking to a 512-px tile drops the latent threshold below 128 so 1024
    (and larger) actually tiles, while preserving the VAE's 8× sample:latent scale."""
    vae = _FakeVae()
    rt.shrink_vae_tiles(vae)
    assert vae.tile_sample_min_size == 512
    assert vae.tile_latent_min_size == 64  # 512 / 8; now 128 (a 1024² latent) > 64 → tiles engage


def test_shrink_vae_tiles_noop_when_already_small() -> None:
    """Never enlarge tiles: a VAE whose tile is already <= 512 is left alone."""
    vae = _FakeVae()
    vae.tile_sample_min_size = 256
    vae.tile_latent_min_size = 32
    rt.shrink_vae_tiles(vae)
    assert vae.tile_sample_min_size == 256
    assert vae.tile_latent_min_size == 32


def test_smaller_resolutions_only_suggests_smaller() -> None:
    """The OOM hint must never suggest a size >= the current one (the reported 512->768/512 bug)."""
    assert rt.smaller_resolutions(1024, 1024) == ["768x768", "512x512"]
    assert rt.smaller_resolutions(512, 512) == ["384x384", "256x256"]
    assert rt.smaller_resolutions(256, 256) == ["128x128"]  # past the ladder -> halve
    # never suggests the current size or larger
    for size in (2048, 1024, 768, 512, 384, 256, 200, 128):
        for s in rt.smaller_resolutions(size, size):
            assert int(s.split("x")[0]) < size
