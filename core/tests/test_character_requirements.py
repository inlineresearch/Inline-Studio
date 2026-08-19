"""What the character nodes need on disk, and that a missing one is reported not swallowed."""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.models.characterreqs import ENCODER_NODES, CharacterEncoderProvider


def test_the_three_encoders_are_declared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """They were only ever a hardcoded list in the client, so nothing else could report them: not
    the model popup, and not a dropped workflow that needs them."""
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    components = CharacterEncoderProvider().components()

    assert [c.id for c in components] == ["yunet", "sface", "dinov2"]
    assert all(c.category == "annotators" for c in components)
    assert all(not c.present for c in components), "nothing is on disk in an empty models root"
    assert all(not c.optional for c in components), "encoding cannot proceed without them"


def test_presence_follows_the_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    annotators = tmp_path / "annotators"
    (annotators / "dinov2-base").mkdir(parents=True)
    (annotators / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
    (annotators / "dinov2-base" / "model.safetensors").write_bytes(b"x")

    present = {c.id: c.present for c in CharacterEncoderProvider().components()}
    assert present == {"yunet": True, "sface": False, "dinov2": True}


def test_dinov2_fetches_named_files_not_the_whole_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It ships a .bin twin beside the safetensors; pulling the folder doubles the download."""
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    dinov2 = next(c for c in CharacterEncoderProvider().components() if c.id == "dinov2")

    assert dinov2.is_folder
    assert "model.safetensors" in dinov2.repo_files
    assert not any(f.endswith(".bin") for f in dinov2.repo_files)


def test_every_encoding_node_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Edit re-encodes nothing today, but it owns the same file and must not claim to need less."""
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    from inline_core.device.memory import MemoryPolicy
    from inline_core.graph.registry import build_default_registry
    from inline_core.models.requirements import RequirementsRegistry
    from inline_core.runtime.file_store import FileTakeStore
    from inline_core.server.bootstrap import register_models

    reqs = RequirementsRegistry()
    register_models(
        build_default_registry(),
        FileTakeStore(tmp_path / "takes"),
        MemoryPolicy(),
        requirements=reqs,
    )

    for node_type in ENCODER_NODES:
        provider = reqs.get(node_type)
        assert provider is not None, f"{node_type} declares no requirements"
        # Constructing them must not raise: an exception here reads to the UI as "needs nothing".
        assert len(provider.components()) == 3
