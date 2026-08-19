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
    """It ships a .bin twin beside the safetensors; pulling the folder doubles the download.

    Those files sit at the repo root, so naming a `repo_folder` as well made the fetch look for a
    subfolder nothing had written: "No such file or directory: .../dinov2-base.part/dinov2-base".
    """
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    dinov2 = next(c for c in CharacterEncoderProvider().components() if c.id == "dinov2")

    assert "model.safetensors" in dinov2.repo_files
    assert not any(f.endswith(".bin") for f in dinov2.repo_files)
    assert not dinov2.repo_folder, "the files are at the repo root, not under a folder"
    assert dinov2.filename == "dinov2-base", "the folder they land in is still named"


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


def test_the_encoders_are_pickable_not_just_visible(tmp_path: Path) -> None:
    """A node that silently uses a file the user cannot see or change is why none of these ever
    showed as missing. Each is a dropdown over models/annotators, defaulting to the shipped one."""
    from inline_core.characters import weights
    from inline_core.models.character.runner import EDIT, ENCODE

    for descriptor in (ENCODE, EDIT):
        params = {p.key: p for p in descriptor.params}
        for key, default in (
            ("face_detector", weights.YUNET_FILE),
            ("face_embedder", weights.SFACE_FILE),
            ("subject_embedder", weights.DINOV2_DIR),
        ):
            assert key in params, f"{descriptor.type} hides {key}"
            assert params[key].options_from == "annotators"
            assert params[key].default == default


def test_a_swapped_encoder_invalidates_the_centroids_it_did_not_produce() -> None:
    """Cosine similarity across two encoders means nothing. The version constants track the
    shipped builds, so a picked file must fold into the identity or a stale centroid reads as
    current."""
    from inline_core.characters import charfile as cf
    from inline_core.characters import scoring

    try:
        scoring.use_encoders()
        shipped = scoring.encoder_versions_by_id()[scoring.DINOV2_ID]
        manifest = cf.Manifest(
            char_id="c", name="Ada", created_at=0, modified_at=0,
            scoring={"encoders": [{"id": scoring.DINOV2_ID, "version": shipped, "dim": 768}]},
        )
        assert cf.centroid_valid(manifest, scoring.DINOV2_ID, shipped)

        scoring.use_encoders(subject_embedder="dinov2-large")
        swapped = scoring.encoder_versions_by_id()[scoring.DINOV2_ID]
        assert swapped != shipped
        assert not cf.centroid_valid(manifest, scoring.DINOV2_ID, swapped)
    finally:
        scoring.use_encoders()


def test_a_picked_encoder_that_is_absent_is_reported_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling back to the shipped file encodes against a different model than the node names."""
    from inline_core.characters import scoring
    from inline_core.errors import ComponentError

    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    try:
        scoring.use_encoders(face_detector="not_here.onnx")
        with pytest.raises(ComponentError, match="not_here.onnx"):
            scoring._encoder_path("yunet", "face_detection_yunet_2023mar.onnx")
    finally:
        scoring.use_encoders()
