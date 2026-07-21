"""The ``load/lora`` handle, the loader cache key it feeds, and the weight fuse itself."""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.errors import ComponentError
from inline_core.graph.loader_runners import ComponentRef, LoadLoraRunner, LoraRef
from inline_core.graph.registry import build_default_registry
from inline_core.graph.schema import Node
from inline_core.models.loaders import _device_key, _dtype_key, lora_cache_key

torch = pytest.importorskip("torch")


# --- the handle -------------------------------------------------------------------------------


def _node(file: str, strength: float = 1.0) -> Node:
    return Node(id="l1", type="load/lora", params={"file": file, "strength": strength})


def _lora_file(root: Path, name: str, keys: dict[str, object]) -> Path:
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    save_file(dict(keys), str(path))  # type: ignore[arg-type]
    return path


def test_load_lora_appends_to_the_stack(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    _lora_file(tmp_path / "loras", "a.safetensors", {"x": torch.zeros(1)})
    upstream = ComponentRef(kind="diffusion", arch="z-image", file="/m.safetensors")

    out = LoadLoraRunner().run(_node("a.safetensors", 0.8), {"model": [upstream]}, ctx=None)  # type: ignore[arg-type]

    ref = out.outputs["model"]
    assert ref.file == upstream.file and ref.kind == "diffusion"  # the model handle is unchanged
    assert len(ref.loras) == 1
    assert ref.loras[0].strength == 0.8


def test_load_lora_chains_in_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    for name in ("a.safetensors", "b.safetensors"):
        _lora_file(tmp_path / "loras", name, {"x": torch.zeros(1)})
    runner = LoadLoraRunner()
    base = ComponentRef(kind="diffusion", arch="z-image", file="/m.safetensors")

    first = runner.run(_node("a.safetensors"), {"model": [base]}, ctx=None).outputs["model"]  # type: ignore[arg-type]
    second = runner.run(_node("b.safetensors"), {"model": [first]}, ctx=None).outputs["model"]  # type: ignore[arg-type]

    assert [Path(x.file).name for x in second.loras] == ["a.safetensors", "b.safetensors"]


def test_load_lora_rejects_a_non_model_input() -> None:
    with pytest.raises(ComponentError, match="diffusion model"):
        LoadLoraRunner().run(_node("a.safetensors"), {"model": [None]}, ctx=None)  # type: ignore[arg-type]


def test_load_lora_is_registered_and_visible() -> None:
    descriptor = build_default_registry().get("load/lora")
    assert descriptor.hidden is False
    assert descriptor.input("model").kind.value == "model"
    assert descriptor.output("model").kind.value == "model"


# --- the cache key ----------------------------------------------------------------------------


def _key(loras: tuple[LoraRef, ...]) -> tuple[str, ...]:
    return ("z-image", "diffusion", "/m.safetensors", _dtype_key("fp16"), "none",
            _device_key("cuda:0")) + lora_cache_key(loras)


def test_no_lora_leaves_the_original_cache_key_shape() -> None:
    """A regression guard: an un-LoRA'd load must key exactly as it did before LoRA existed."""
    assert len(_key(())) == 6


def test_different_stacks_never_share_a_cache_entry() -> None:
    a = _key((LoraRef("/a.safetensors", 1.0),))
    b = _key((LoraRef("/b.safetensors", 1.0),))
    same_file_other_strength = _key((LoraRef("/a.safetensors", 0.5),))
    reordered = _key((LoraRef("/b.safetensors", 1.0), LoraRef("/a.safetensors", 1.0)))
    stacked = _key((LoraRef("/a.safetensors", 1.0), LoraRef("/b.safetensors", 1.0)))

    assert len({_key(()), a, b, same_file_other_strength, reordered, stacked}) == 6
    assert a == _key((LoraRef("/a.safetensors", 1.0),))  # deterministic


def test_switching_lora_stacks_evicts_the_other_variant() -> None:
    """The bug this guards: eviction matched on file path only, so fusing a LoRA into an
    already-loaded checkpoint kept the unfused transformer AND built a fused one - two full-size
    models resident, which OOMs a 16 GB card. Same file, different stack => the old one must go."""
    from inline_core.models import loaders

    loaders._CACHE.clear()
    unfused = ("z-image", "diffusion", "/m.safetensors", "fp16", "none", "cuda:0")
    fused = unfused + lora_cache_key((LoraRef("/a.safetensors", 1.0),))
    vae = ("z-image", "vae", "/ae.safetensors", "fp16", "none", "cuda:0")
    loaders._CACHE.update({unfused: object(), fused: object(), vae: object()})

    # About to load the fused variant of the same file: the unfused one is now stale.
    loaders.unload_components(
        keep_files={"/m.safetensors", "/ae.safetensors"}, keep_loras=fused[6:]
    )

    assert unfused not in loaders._CACHE
    assert fused in loaders._CACHE
    assert vae in loaders._CACHE  # a non-diffusion component of a kept file is untouched
    loaders._CACHE.clear()


def test_unload_without_keep_loras_is_unchanged() -> None:
    """Callers that never pass keep_loras keep the original file-only eviction semantics."""
    from inline_core.models import loaders

    loaders._CACHE.clear()
    keep = ("z-image", "diffusion", "/m.safetensors", "fp16", "none", "cuda:0")
    drop = ("z-image", "diffusion", "/other.safetensors", "fp16", "none", "cuda:0")
    loaders._CACHE.update({keep: object(), drop: object()})

    loaders.unload_components(keep_files={"/m.safetensors"})

    assert keep in loaders._CACHE and drop not in loaders._CACHE
    loaders._CACHE.clear()


# --- the fuse ---------------------------------------------------------------------------------


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)


def test_fuse_applies_scaled_delta(tmp_path) -> None:
    from inline_core.models.lora import fuse_loras

    model = _Tiny()
    torch.nn.init.zeros_(model.proj.weight)
    down = torch.ones(2, 4)
    up = torch.ones(4, 2)
    path = _lora_file(
        tmp_path, "l.safetensors",
        {"proj.lora_down.weight": down, "proj.lora_up.weight": up},
    )

    fuse_loras(model, (LoraRef(str(path), 0.5),))

    # up @ down is all-2s (rank 2); at strength 0.5 every weight lands on 1.0.
    assert torch.allclose(model.proj.weight, torch.ones(4, 4))


def test_fuse_honours_kohya_alpha(tmp_path) -> None:
    from inline_core.models.lora import fuse_loras

    model = _Tiny()
    torch.nn.init.zeros_(model.proj.weight)
    path = _lora_file(
        tmp_path, "l.safetensors",
        {
            "proj.lora_down.weight": torch.ones(2, 4),
            "proj.lora_up.weight": torch.ones(4, 2),
            "proj.alpha": torch.tensor(1.0),  # alpha/rank = 1/2
        },
    )

    fuse_loras(model, (LoraRef(str(path), 1.0),))

    assert torch.allclose(model.proj.weight, torch.ones(4, 4))


def test_fuse_strips_a_diffusion_model_prefix(tmp_path) -> None:
    from inline_core.models.lora import fuse_loras

    model = _Tiny()
    torch.nn.init.zeros_(model.proj.weight)
    path = _lora_file(
        tmp_path, "l.safetensors",
        {
            "diffusion_model.proj.lora_A.weight": torch.ones(2, 4),
            "diffusion_model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )

    fuse_loras(model, (LoraRef(str(path), 1.0),))

    assert torch.allclose(model.proj.weight, torch.full((4, 4), 2.0))


def test_fuse_fails_loudly_on_an_unmatched_layer(tmp_path) -> None:
    """A partial apply degrades output without erroring - the one failure mode worth being noisy
    about, so any unmatched key is fatal rather than a warning."""
    from inline_core.models.lora import fuse_loras

    path = _lora_file(
        tmp_path, "l.safetensors",
        {
            "proj.lora_down.weight": torch.ones(2, 4),
            "proj.lora_up.weight": torch.ones(4, 2),
            "not_a_layer.lora_down.weight": torch.ones(2, 4),
            "not_a_layer.lora_up.weight": torch.ones(4, 2),
        },
    )

    with pytest.raises(ComponentError, match="does not match this model"):
        fuse_loras(_Tiny(), (LoraRef(str(path), 1.0),))


def test_fuse_of_an_empty_stack_is_a_noop() -> None:
    from inline_core.models.lora import fuse_loras

    model = _Tiny()
    before = model.proj.weight.clone()

    fuse_loras(model, ())

    assert torch.equal(model.proj.weight, before)
