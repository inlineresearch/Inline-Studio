"""The load/* runners - torch-free file resolution into a typed ComponentRef handle. No weights are
read; the runner just points at a file the consuming model runner later loads."""

from __future__ import annotations

import pytest

from inline_core.errors import ComponentError
from inline_core.graph.loader_runners import ComponentRef, register_loaders
from inline_core.graph.registry import Registry, build_default_registry
from inline_core.graph.schema import Node
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import CollectingEmitter


def _ctx() -> ExecutionContext:
    from inline_core.device.memory import MemoryPolicy

    return ExecutionContext(
        run_id="r", policy=MemoryPolicy(), emitter=CollectingEmitter(), cancel=CancelToken()
    )


def _models_root(monkeypatch, tmp_path):
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    return tmp_path


def test_loaders_registered_visible_with_runners() -> None:
    reg = build_default_registry()
    for node_type, port in (
        ("load/diffusion-model", "model"),
        ("load/vae", "vae"),
        ("load/text-encoder", "text_encoder"),
    ):
        assert reg.get(node_type).hidden is False
        assert reg.get(node_type).output(port) is not None
        assert reg.runner(node_type) is not None


def test_explicit_file_resolves_to_ref(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "ae.safetensors").write_bytes(b"")
    reg = Registry()
    register_loaders(reg)
    node = Node(id="v", type="load/vae", params={"file": "ae.safetensors"})
    result = reg.runner("load/vae").run(node, {}, _ctx())
    ref = result.outputs["vae"]
    assert isinstance(ref, ComponentRef)
    assert ref.kind == "vae" and ref.arch == "z-image"
    assert ref.file == str(root / "vae" / "ae.safetensors")


def test_auto_picks_the_single_file(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / "z_image_bf16.safetensors").write_bytes(b"")
    reg = Registry()
    register_loaders(reg)
    node = Node(id="m", type="load/diffusion-model", params={})  # no explicit file -> auto
    ref = reg.runner("load/diffusion-model").run(node, {}, _ctx()).outputs["model"]
    assert ref.kind == "diffusion"
    assert ref.file.endswith("z_image_bf16.safetensors")


def test_missing_file_raises(monkeypatch, tmp_path) -> None:
    _models_root(monkeypatch, tmp_path)  # empty root
    reg = Registry()
    register_loaders(reg)
    node = Node(id="v", type="load/vae", params={})
    with pytest.raises(ComponentError, match="No model file"):
        reg.runner("load/vae").run(node, {}, _ctx())


def test_selected_but_absent_file_raises(monkeypatch, tmp_path) -> None:
    root = _models_root(monkeypatch, tmp_path)
    (root / "vae").mkdir(parents=True)
    reg = Registry()
    register_loaders(reg)
    node = Node(id="v", type="load/vae", params={"file": "nope.safetensors"})
    with pytest.raises(ComponentError, match="not found"):
        reg.runner("load/vae").run(node, {}, _ctx())
