"""What a wired ``load/*`` node hands the LTX node, and how the node reads it back.

Both of these shipped broken past a green suite, and neither raised where it went wrong: the LoRA
one only failed once the models root was relative, and the component one failed silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.errors import ComponentError
from inline_core.graph.loader_runners import ComponentRef, LoraRef
from inline_core.models.ltx25 import pipeline


def test_a_wired_component_is_read_off_the_field_the_loader_actually_sets(tmp_path: Path) -> None:
    """`ComponentRef` carries `file`. Reading `path` matched nothing, so a wired transformer, VAE
    or text encoder was silently dropped and the run used the dropdown instead."""
    weights = tmp_path / "custom-transformer.safetensors"
    weights.write_bytes(b"")
    ref = ComponentRef(kind="diffusion", arch="ltx-2-5", file=str(weights))

    assert pipeline._wired(ref) == weights


def test_an_unwired_port_resolves_to_nothing() -> None:
    assert pipeline._wired(None) is None


def test_a_lora_is_taken_at_the_path_the_loader_resolved(tmp_path: Path, monkeypatch) -> None:
    """The models root is relative by default (`./models`), which is what exposed this: the ref's
    path was re-joined under `models/loras/`, so `models/loras/x.safetensors` was looked for at
    `models/loras/models/loras/x.safetensors`. An absolute root had hidden it, because an absolute
    right-hand side wins a join."""
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    loras = tmp_path / "models" / "loras"
    loras.mkdir(parents=True)
    adapter = loras / "motion-8afbbc83.safetensors"
    adapter.write_bytes(b"")
    monkeypatch.setenv("INLINE_MODELS_DIR", "models")

    # Exactly what LoadLoraRunner emits under a relative root.
    emitted = LoraRef(file="models/loras/motion-8afbbc83.safetensors", strength=0.8)
    resolved = pipeline._lora(emitted)

    assert Path(resolved.path) == Path("models/loras/motion-8afbbc83.safetensors")
    assert resolved.strength == pytest.approx(0.8)


def test_a_lora_that_is_not_there_names_itself_in_the_error() -> None:
    with pytest.raises(ComponentError, match="ghost.safetensors"):
        pipeline._lora(LoraRef(file="models/loras/ghost.safetensors", strength=1.0))
