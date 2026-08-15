"""char_encode and char_apply: normalisation, payload invalidation, and reference ordering.

The parts that need the scoring weights are gated, so the suite still runs offline. What is left
covers the logic that decides what a render actually sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from inline_core.characters import apply as ap  # noqa: E402
from inline_core.characters import charfile as cf  # noqa: E402
from inline_core.characters import encode, library  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)


def _image(path: Path, size: tuple[int, int], colour: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, colour).save(path)
    return path


# --- normalisation -------------------------------------------------------------------------------


def test_normalise_snaps_onto_the_grid_and_into_the_budget() -> None:
    out = encode.normalise_reference(Image.new("RGB", (3000, 2000)))
    assert out.width % 16 == 0 and out.height % 16 == 0
    assert out.width * out.height <= encode.PAYLOAD_POLICY["max_pixels"]
    # Aspect survives the resize, or the reference reframes the subject.
    assert abs((out.width / out.height) - 1.5) < 0.02


def test_normalise_leaves_an_already_conforming_image_alone() -> None:
    original = Image.new("RGB", (512, 512))
    assert encode.normalise_reference(original) is original


def test_normalise_never_returns_a_zero_dimension() -> None:
    out = encode.normalise_reference(Image.new("RGB", (8, 8)))
    assert out.width >= 16 and out.height >= 16


# --- hints ---------------------------------------------------------------------------------------


def test_one_image_asks_for_exactly_one_thing() -> None:
    """Another angle and a profile view are the same request worded twice."""
    assert encode.strength_hints([(512, 512)]) == ["Add a second angle"]


def test_hints_escalate_rather_than_stack() -> None:
    square, wide, tall = (512, 512), (768, 512), (512, 768)
    assert encode.strength_hints([square, wide]) == ["Add a profile view"]
    assert encode.strength_hints([square, wide, tall]) == ["Add a different outfit or setting"]
    assert encode.strength_hints([square, wide, tall, wide]) == []


def test_uniformly_cropped_refs_ask_for_a_different_crop() -> None:
    assert encode.strength_hints([(512, 512)] * 4) == ["Add a wider or tighter crop"]


def test_hints_are_recomputed_from_the_manifest_not_read_back() -> None:
    """A rule change must reach characters already on disk, not just newly encoded ones."""
    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    manifest.refs.append({"path": "refs/000.png", "sha256": "x", "width": 640, "height": 480})
    manifest.hints = ["Something the old rules said"]
    assert encode.hints_for(manifest) == ["Add a second angle"]


# --- payload invalidation ------------------------------------------------------------------------


def _manifest_with_refs(count: int) -> tuple[cf.Manifest, dict[str, bytes], list[Image.Image]]:
    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    members: dict[str, bytes] = {}
    images: list[Image.Image] = []
    for index in range(count):
        image = Image.new("RGB", (640, 480), (index * 40, 90, 140))
        images.append(image)
        member = cf.member_name("refs", index, ".png")
        data = encode._png_bytes(image)
        members[member] = data
        manifest.refs.append({"path": member, "sha256": cf.sha256_bytes(data)})
    return manifest, members, images


def test_build_payload_is_valid_and_replaces_a_previous_one() -> None:
    manifest, members, images = _manifest_with_refs(2)
    encode.build_payload(manifest, members, images)
    assert cf.payload_valid(manifest, encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_ENCODER_VERSION)

    # A stale member from an older compile must not survive a rebuild.
    members[f"payloads/{encode.FLUX2_KLEIN_ARCH}/ref_009.png"] = b"stale"
    encode.build_payload(manifest, members, images)
    assert f"payloads/{encode.FLUX2_KLEIN_ARCH}/ref_009.png" not in members


def test_editing_a_ref_invalidates_the_payload() -> None:
    manifest, members, images = _manifest_with_refs(2)
    encode.build_payload(manifest, members, images)
    manifest.refs[0]["sha256"] = cf.sha256_bytes(b"replaced")
    assert not cf.payload_valid(manifest, encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_ENCODER_VERSION)


def test_a_bumped_compiler_version_invalidates_the_payload() -> None:
    manifest, members, images = _manifest_with_refs(1)
    encode.build_payload(manifest, members, images)
    assert not cf.payload_valid(manifest, encode.FLUX2_KLEIN_ARCH, "999")


# --- char_apply ----------------------------------------------------------------------------------


def _saved_character(name: str = "Ada", description: str = "", refs: int = 2) -> Path:
    manifest, members, images = _manifest_with_refs(refs)
    manifest.name = name
    members["text/description.md"] = description.encode()
    manifest.text = {"path": "text/description.md", "sha256": cf.sha256_bytes(description.encode())}
    encode.build_payload(manifest, members, images)
    return library.save(cf.CharDoc(manifest=manifest, members=members))


def test_char_apply_returns_the_payload_refs_in_order() -> None:
    _saved_character(refs=3)
    applied = ap.char_apply("Ada.char")
    assert applied is not None
    assert [Path(r.path).name for r in applied.refs] == [
        "ref_000.png", "ref_001.png", "ref_002.png"
    ]
    assert all(Path(r.path).is_file() for r in applied.refs)


def test_no_pick_applies_nothing() -> None:
    assert ap.char_apply("") is None
    assert ap.char_apply(None) is None  # type: ignore[arg-type]


def test_a_missing_character_raises_rather_than_generating_without_it() -> None:
    """Silently dropping the character produces a plausible image of the wrong person, which is the
    exact failure this feature exists to prevent."""
    with pytest.raises(FileNotFoundError, match="Nobody.char"):
        ap.char_apply("Nobody.char")


def test_prompt_prefix_names_the_positions_the_refs_will_occupy() -> None:
    _saved_character(description="short brown hair", refs=2)
    applied = ap.char_apply("Ada.char")
    assert applied is not None
    # Two user-wired refs already hold 1 and 2, so the character lands at 3 and 4.
    prefix = applied.prompt_prefix(3)
    assert "Images 3 and 4 show Ada" in prefix
    assert "short brown hair" in prefix
    assert prefix.endswith(" ")


def test_prompt_prefix_is_singular_for_one_reference() -> None:
    _saved_character(refs=1)
    applied = ap.char_apply("Ada.char")
    assert applied is not None
    assert applied.prompt_prefix(1).startswith("Image 1 shows Ada")


def test_prompt_prefix_without_a_description_is_still_well_formed() -> None:
    _saved_character(description="", refs=1)
    applied = ap.char_apply("Ada.char")
    assert applied is not None
    assert applied.prompt_prefix(2) == "Image 2 shows Ada, the same character in every image. "


def test_a_stale_payload_is_recompiled_from_refs_and_rewritten() -> None:
    path = _saved_character(refs=2)
    doc = cf.read(path)
    doc.manifest.payloads[encode.FLUX2_KLEIN_ARCH]["encoder"]["version"] = "0"
    cf.write(path, doc)

    applied = ap.char_apply("Ada.char")
    assert applied is not None
    assert len(applied.refs) == 2
    rebuilt = cf.read(path)
    assert cf.payload_valid(
        rebuilt.manifest, encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_ENCODER_VERSION
    )


def test_editing_a_character_extracts_to_a_new_directory() -> None:
    """The cache is keyed by content, so a running graph never reads a directory being rewritten."""
    path = _saved_character(refs=1)
    first = ap.char_apply("Ada.char")
    assert first is not None

    manifest, members, images = _manifest_with_refs(1)
    manifest.name = "Ada"
    members["refs/000.png"] = encode._png_bytes(Image.new("RGB", (640, 480), (7, 7, 7)))
    manifest.refs[0]["sha256"] = cf.sha256_bytes(members["refs/000.png"])
    encode.build_payload(manifest, members, [Image.new("RGB", (640, 480), (7, 7, 7))])
    cf.write(path, cf.CharDoc(manifest=manifest, members=members))

    second = ap.char_apply("Ada.char")
    assert second is not None
    assert Path(first.refs[0].path).parent != Path(second.refs[0].path).parent


# --- full encode, needs the scoring weights ------------------------------------------------------


@pytest.mark.skipif(
    not Path.home().joinpath(".cache/huggingface").is_dir(),
    reason="no HF cache for scoring weights",
)
def test_char_encode_builds_every_section(tmp_path: Path) -> None:
    src = _image(tmp_path / "a.png", (768, 512), (120, 90, 60))
    doc = encode.char_encode([src], name="Ada", description="green jacket")

    assert len(doc.manifest.refs) == 1
    assert doc.manifest.text["path"] == "text/description.md"
    assert doc.members["text/description.md"] == b"green jacket"
    assert cf.payload_valid(
        doc.manifest, encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_ENCODER_VERSION
    )
    # A flat colour has no face, so the character is subject-scored only.
    assert doc.manifest.scoring["face_bearing"] is False
    assert doc.manifest.hints == ["Add a second angle"]


def test_char_encode_rejects_an_empty_or_missing_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        encode.char_encode([], name="Ada")
    with pytest.raises(ValueError, match="not found"):
        encode.char_encode([tmp_path / "nope.png"], name="Ada")
