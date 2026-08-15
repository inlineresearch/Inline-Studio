"""The .char container: round-trip, signature placement, and the invalidation rules."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from inline_core.characters import charfile as cf

_POLICY = {"max_pixels": 1048576, "multiple_of": 16}


def _doc(name: str = "Ada") -> cf.CharDoc:
    refs = {
        cf.member_name("refs", 0, ".png"): b"first-ref-bytes",
        cf.member_name("refs", 1, ".png"): b"second-ref-bytes",
    }
    manifest = cf.Manifest(
        char_id="7f1c0d2e-0000-4000-8000-000000000001",
        name=name,
        created_at=1755000000,
        modified_at=1755000000,
        app_version="1.2.70",
        refs=[
            {"path": path, "sha256": cf.sha256_bytes(data), "width": 512, "height": 512}
            for path, data in refs.items()
        ],
        text={"path": "text/description.md", "sha256": cf.sha256_bytes(b"# Ada\n")},
    )
    manifest.payloads["flux2-klein"] = {
        "payload_version": 1,
        "encoder": {"id": "flux2-klein-refset", "version": "1"},
        "source_sha256": cf.refs_fingerprint(manifest, _POLICY),
        "policy": _POLICY,
        "files": [{"path": "payloads/flux2-klein/ref_000.png", "sha256": "0" * 64}],
    }
    manifest.scoring = {
        "encoders": [
            {"id": "sface", "version": "2021dec", "dim": 128},
            {"id": "dinov2-base", "version": "1", "dim": 768},
        ],
        "centroids": {"sface": "scoring/centroid_sface.json"},
        "face_bearing": True,
        "blend": {"face": 0.7, "subject": 0.3},
    }
    members = dict(refs)
    members["text/description.md"] = b"# Ada\n"
    members["payloads/flux2-klein/ref_000.png"] = b"normalised"
    members["scoring/centroid_sface.json"] = b'{"vector":[0.1],"count":2}'
    return cf.CharDoc(manifest=manifest, members=members)


def test_round_trip_preserves_manifest_and_bytes(tmp_path: Path) -> None:
    original = _doc()
    path = cf.write(tmp_path / "Ada.char", original)
    loaded = cf.read(path)
    assert loaded.manifest.to_json() == original.manifest.to_json()
    assert loaded.members == original.members


def test_signature_is_at_byte_zero_of_the_manifest(tmp_path: Path) -> None:
    path = cf.write(tmp_path / "Ada.char", _doc())
    with zipfile.ZipFile(path) as archive:
        # The manifest is written first, so it is also the first member in the archive.
        assert archive.namelist()[0] == cf.MANIFEST_NAME
        raw = archive.read(cf.MANIFEST_NAME)
    assert raw.startswith(cf.SIGNATURE)
    assert json.loads(raw)["magic"] == cf.MAGIC
    assert cf.looks_like_char(raw)


def test_mutating_a_ref_invalidates_the_payload(tmp_path: Path) -> None:
    doc = _doc()
    assert cf.payload_valid(doc.manifest, "flux2-klein", "1")
    doc.manifest.refs[0]["sha256"] = cf.sha256_bytes(b"a different image")
    assert not cf.payload_valid(doc.manifest, "flux2-klein", "1")


def test_reordering_refs_invalidates_the_payload() -> None:
    """Ordinal prompting makes ref order meaning, so a reorder is a different payload."""
    doc = _doc()
    doc.manifest.refs.reverse()
    assert not cf.payload_valid(doc.manifest, "flux2-klein", "1")


def test_bumped_encoder_version_invalidates_the_payload() -> None:
    doc = _doc()
    assert not cf.payload_valid(doc.manifest, "flux2-klein", "2")


def test_unknown_arch_has_no_payload() -> None:
    assert not cf.payload_valid(_doc().manifest, "h3-video", "1")


def test_centroid_valid_tracks_encoder_version() -> None:
    manifest = _doc().manifest
    assert cf.centroid_valid(manifest, "sface", "2021dec")
    assert not cf.centroid_valid(manifest, "sface", "2024mar")
    assert not cf.centroid_valid(manifest, "arcface", "1")


def test_a_newer_format_version_refuses_to_load(tmp_path: Path) -> None:
    path = tmp_path / "future.char"
    raw = _doc().manifest.to_json()
    raw["format_version"] = cf.FORMAT_VERSION + 1
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(cf.MANIFEST_NAME, json.dumps(raw, separators=(",", ":")))
    with pytest.raises(cf.CharFileError, match="newer version"):
        cf.read(path)


def test_a_non_char_zip_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.char"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(cf.CharFileError, match="no manifest"):
        cf.read(path)


def test_a_non_zip_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.char"
    path.write_bytes(b"not a zip at all")
    with pytest.raises(cf.CharFileError):
        cf.read(path)


def test_a_failed_write_leaves_no_staging_file_and_keeps_the_previous(tmp_path: Path) -> None:
    path = tmp_path / "Ada.char"
    cf.write(path, _doc(name="Ada"))
    before = path.read_bytes()

    # Fails after the manifest is already in the staging archive, which is the case that would
    # otherwise leave a half-written .char behind.
    broken = _doc()
    broken.members["payloads/flux2-klein/ref_000.png"] = object()  # type: ignore[assignment]
    with pytest.raises(TypeError):
        cf.write(path, broken)

    assert not (tmp_path / "Ada.char.part").exists()
    assert path.read_bytes() == before


def test_member_names_stay_ascii_for_a_non_ascii_character_name(tmp_path: Path) -> None:
    doc = _doc(name="Ada Løvelace 天")
    path = cf.write(tmp_path / "Ada.char", doc)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    assert all(name.isascii() and "\\" not in name for name in names)
    assert cf.read(path).manifest.name == "Ada Løvelace 天"


def test_writing_twice_produces_the_same_manifest_bytes(tmp_path: Path) -> None:
    """A fixed key order means an unchanged character does not churn its bytes."""
    doc = _doc()
    first = cf.write(tmp_path / "a.char", doc)
    second = cf.write(tmp_path / "b.char", doc)
    with zipfile.ZipFile(first) as a, zipfile.ZipFile(second) as b:
        assert a.read(cf.MANIFEST_NAME) == b.read(cf.MANIFEST_NAME)
