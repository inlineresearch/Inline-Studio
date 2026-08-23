"""char_encode and char_apply: normalisation, payload invalidation, and reference ordering.

The parts that need the scoring weights are gated, so the suite still runs offline. What is left
covers the logic that decides what a render actually sees.
"""

from __future__ import annotations

import io
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


# --- payload kinds --------------------------------------------------------------------------------


def test_a_reference_payload_keeps_the_bare_arch_key() -> None:
    """Every .char written before there was a second kind must stay valid without a format bump."""
    assert encode.payload_key("flux2-klein") == "flux2-klein"
    assert encode.payload_key("flux2-klein", encode.PAYLOAD_REF) == "flux2-klein"


def test_a_lora_payload_sits_beside_it_rather_than_replacing_it() -> None:
    assert encode.payload_key("flux2-klein", encode.PAYLOAD_LORA) == "flux2-klein-lora"
    assert encode.payload_key("krea2", encode.PAYLOAD_LORA) == "krea2-lora"


def test_a_stored_adapter_records_the_base_it_was_trained_against(tmp_path: Path) -> None:
    """A 4B adapter on a 9B degrades silently rather than raising, so the base has to travel."""
    doc = encode.char_encode([_image(tmp_path / "r.png", (768, 1024), (180, 150, 140))], name="Ada")
    key = encode.set_lora_payload(
        doc.manifest, doc.members, b"adapter-bytes",
        base="flux-2-klein-base-4b.safetensors", rank=16, steps=200, resolution=512,
    )

    entry = doc.manifest.payloads[key]
    assert entry["type"] == encode.PAYLOAD_LORA
    assert entry["base"] == "flux-2-klein-base-4b.safetensors"
    assert entry["training"] == {"rank": 16, "steps": 200, "resolution": 512}
    assert doc.members[f"payloads/{key}/adapter.safetensors"] == b"adapter-bytes"
    # The reference payload is untouched: compiling for one model never edits another.
    assert doc.manifest.payloads[encode.FLUX2_KLEIN_ARCH]["type"] == encode.PAYLOAD_REF


def test_editing_the_references_invalidates_a_stored_adapter(tmp_path: Path) -> None:
    """An adapter trained on an old reference set is the wrong face, not a staler one."""
    doc = encode.char_encode([_image(tmp_path / "r.png", (768, 1024), (180, 150, 140))], name="Ada")
    key = encode.set_lora_payload(
        doc.manifest, doc.members, b"adapter-bytes",
        base="flux-2-klein-base-4b.safetensors", rank=16, steps=200, resolution=512,
    )
    assert cf.payload_valid(doc.manifest, key, encode.LORA_PAYLOAD_VERSION)

    doc.manifest.refs.append({**doc.manifest.refs[0], "sha256": "0" * 64})

    assert not cf.payload_valid(doc.manifest, key, encode.LORA_PAYLOAD_VERSION)


def test_an_encode_reports_its_phases_in_order(tmp_path: Path) -> None:
    """Phases must arrive, advance and finish at 1.0; a silent encode looks like a hang."""
    seen: list[tuple[float, str]] = []
    encode.char_encode(
        [
            _image(tmp_path / "a.png", (768, 1024), (180, 150, 140)),
            _image(tmp_path / "b.png", (768, 1024), (170, 140, 130)),
        ],
        name="Ada",
        on_progress=lambda fraction, status: seen.append((fraction, status)),
    )

    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions), f"progress went backwards: {fractions}"
    assert fractions[-1] == 1.0, "the encode never reported completion"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    # Every phase names what it is doing; a blank status is what the bar exists to avoid.
    assert all(status.strip() for _, status in seen)
    # Per-reference phases, so a slow multi-ref encode still moves.
    assert sum("Finding faces" in status for _, status in seen) == 2


def test_one_character_holds_a_payload_per_model(tmp_path: Path) -> None:
    """Building for a second model must not disturb the first: identity is compiled once and each
    model gets its own payload, which is what makes adding a model cheap."""
    doc = encode.char_encode([_image(tmp_path / "r.png", (768, 1024), (180, 150, 140))], name="Ada")
    manifest, members = doc.manifest, doc.members

    # A reference set for a second architecture, alongside FLUX.2's.
    encode.build_payload(manifest, members, [encode._open(tmp_path / "r.png")], arch="minimax-h3")
    encode.set_lora_payload(
        manifest, members, b"adapter", arch="krea2",
        base="krea2_turbo_bf16.safetensors", rank=16, steps=600, resolution=512,
    )

    assert set(manifest.payloads) == {encode.FLUX2_KLEIN_ARCH, "minimax-h3", "krea2-lora"}
    assert encode.needs_rebuild(manifest) is False, "nothing was edited, so nothing is stale"
    # Each payload keeps its own files rather than sharing FLUX.2's.
    assert any(m.startswith("payloads/minimax-h3/") for m in members)


def test_a_changed_reference_stales_every_payload(tmp_path: Path) -> None:
    """The fingerprint is shared, so editing references invalidates every model at once - which is
    what stops a stale adapter rendering the wrong face on one model but not another."""
    doc = encode.char_encode([_image(tmp_path / "r.png", (768, 1024), (180, 150, 140))], name="Ada")
    encode.set_lora_payload(
        doc.manifest, doc.members, b"adapter", arch="krea2",
        base="krea2_turbo_bf16.safetensors", rank=16, steps=600, resolution=512,
    )
    assert encode.stale_payloads(doc.manifest) == []

    doc.manifest.refs.append({**doc.manifest.refs[0], "sha256": "0" * 64})

    assert sorted(encode.stale_payloads(doc.manifest)) == ["flux2-klein", "krea2-lora"]
    assert encode.needs_rebuild(doc.manifest) is True


def test_a_payload_is_judged_against_the_policy_it_was_built_with(tmp_path: Path) -> None:
    """Two models normalise references differently, so a fingerprint only means something beside
    the policy that produced it. Judging against the current default would call it stale forever."""
    doc = encode.char_encode([_image(tmp_path / "r.png", (768, 1024), (180, 150, 140))], name="Ada")
    entry = doc.manifest.payloads[encode.FLUX2_KLEIN_ARCH]
    entry["policy"] = {"max_pixels": 512 * 512, "multiple_of": 32}

    assert encode.payload_stale(doc.manifest, encode.FLUX2_KLEIN_ARCH) is True, (
        "a payload built under another policy no longer matches this reference set"
    )


# --- origins, the harvested pool, and what must survive it ---------------------------------------


def _harvest(doc: cf.CharDoc, colour: tuple[int, int, int] = (10, 200, 30)) -> None:
    """Add a harvested reference without running an encoder: only the bookkeeping is under test."""
    image = Image.new("RGB", (512, 512), colour)
    member = cf.member_name("harvested", len(encode.harvested(doc.manifest)), ".png")
    data = encode._png_bytes(image)
    doc.members[member] = data
    doc.manifest.refs.append(
        {
            "path": member,
            "sha256": cf.sha256_bytes(data),
            "width": image.width,
            "height": image.height,
            "origin": cf.ORIGIN_HARVESTED,
        }
    )


def test_a_reference_with_no_origin_is_an_original() -> None:
    """Every character written before harvesting existed has references and no origin field."""
    manifest, _members, _images = _manifest_with_refs(2)
    assert len(encode.originals(manifest)) == 2
    assert encode.harvested(manifest) == []


def test_harvesting_does_not_invalidate_a_trained_adapter() -> None:
    """The fingerprint covers originals only. A harvested reference in it would mark the adapter
    stale, `_extract_lora` drops a stale adapter with an INFO log, and the loop whose whole point
    is a better adapter would silently switch off the one the user has."""
    manifest, members, images = _manifest_with_refs(3)
    encode.build_payload(manifest, members, images)
    encode.set_lora_payload(
        manifest, members, b"ADAPTER", base="flux2-klein-4b", rank=16, steps=500, resolution=512
    )
    doc = cf.CharDoc(manifest=manifest, members=members)
    key = encode.payload_key(encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_LORA)
    assert cf.payload_valid(manifest, key, encode.LORA_PAYLOAD_VERSION)

    _harvest(doc)

    assert cf.payload_valid(manifest, key, encode.LORA_PAYLOAD_VERSION), "the adapter went stale"
    assert cf.payload_valid(manifest, encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_ENCODER_VERSION)


def test_dropping_or_adding_an_original_still_invalidates() -> None:
    manifest, members, images = _manifest_with_refs(3)
    encode.build_payload(manifest, members, images)
    doc = cf.CharDoc(manifest=manifest, members=members)
    encode.drop_ref(doc, 0)
    assert encode.payload_stale(manifest, encode.FLUX2_KLEIN_ARCH)


def test_a_recompile_takes_harvested_references_in_behind_the_originals(tmp_path: Path) -> None:
    """Position is meaning - FLUX.2 addresses a reference by number - so the ones the user vouched
    for hold the leading slots however the manifest happens to be ordered."""
    manifest, members, _images = _manifest_with_refs(2)
    doc = cf.CharDoc(manifest=manifest, members=members)
    _harvest(doc)
    # An original added after the harvest lands at the end of `manifest.refs`, not before it.
    encode.append_refs(doc, [_image(tmp_path / "late.png", (64, 64), (1, 2, 3))])
    assert [cf.origin_of(r) for r in manifest.refs] == [
        cf.ORIGIN_ORIGINAL, cf.ORIGIN_ORIGINAL, cf.ORIGIN_HARVESTED, cf.ORIGIN_ORIGINAL
    ]

    encode.build_payload(manifest, members, encode.ref_images(doc))

    entry = manifest.payloads[encode.FLUX2_KLEIN_ARCH]
    assert entry["harvested_count"] == 1
    assert len(entry["files"]) == 4
    # The harvested one is last in the payload, whatever position it holds in the manifest.
    sizes = [Image.open(io.BytesIO(members[f["path"]])).size for f in entry["files"]]
    assert sizes[3] == (512, 512), "the harvested reference did not land in the last slot"


def test_hints_count_the_originals_only() -> None:
    """Otherwise harvesting silences the very prompts - another angle, a full-body shot - that the
    pool depends on being met."""
    manifest, members, _images = _manifest_with_refs(1)
    doc = cf.CharDoc(manifest=manifest, members=members)
    assert encode.hints_for(manifest) == ["Add a second angle"]
    _harvest(doc)
    _harvest(doc, (200, 10, 30))
    assert encode.hints_for(manifest) == ["Add a second angle"]


def test_the_harvest_cap_never_lets_the_pool_outgrow_the_originals() -> None:
    manifest, _members, _images = _manifest_with_refs(3)
    assert encode.harvest_cap(manifest) == 3
    manifest.refs = manifest.refs[:1]
    assert encode.harvest_cap(manifest) == 1


def test_changing_the_reference_set_marks_scoring_for_a_rebuild() -> None:
    """`drop_ref` leaves the stored per-reference lists describing a set that no longer exists, so
    scoring kept best-matching a take against the reference just deleted for being the wrong
    person. The lists are compacted, so there is no index surgery available - it has to go."""
    manifest, members, _images = _manifest_with_refs(3)
    manifest.scoring = {"refFramings": [0.1, 0.1, 0.1], "refCount": 3, "originals": {"refs": []}}
    members["scoring/embeds_sface.json"] = b'{"vectors":[]}'
    doc = cf.CharDoc(manifest=manifest, members=members)

    encode.drop_ref(doc, 1)

    assert "refFramings" not in manifest.scoring, "a phantom reference survived in scoring"
    assert "scoring/embeds_sface.json" not in members
    # The frozen identity is not derived from the set that changed, so it is not collateral.
    assert "originals" in manifest.scoring


def test_decoding_references_never_returns_a_short_list() -> None:
    """Every scoring position is an index into `manifest.refs`. A skipped member would shorten the
    list and shift each position after it onto a different reference, silently."""
    manifest, members, _images = _manifest_with_refs(3)
    doc = cf.CharDoc(manifest=manifest, members=members)
    assert len(encode.ref_images(doc)) == 3

    members.pop(str(manifest.refs[1]["path"]))
    with pytest.raises(cf.CharFileError, match="missing"):
        encode.ref_images(doc)


def test_pruning_drops_the_harvested_reference_that_adds_least(monkeypatch) -> None:
    """Coverage, not score: a pool of near-duplicates of the best-scoring angle is worth less to a
    compile or a train than one spanning the angles the originals miss."""
    from inline_core.characters import scoring

    manifest, members, _images = _manifest_with_refs(2)
    doc = cf.CharDoc(manifest=manifest, members=members)
    # Two originals sitting on one axis; a near-duplicate of them, and a genuinely new angle.
    frozen = {"refs/000.png": [1.0, 0.0, 0.0], "refs/001.png": [0.99, 0.1, 0.0]}
    members["scoring/originals_dinov2-base.json"] = scoring.dump_keyed(frozen)
    manifest.scoring["originals"] = {"refs": [{"path": p, "sha256": ""} for p in frozen]}
    _harvest(doc, (10, 200, 30))
    _harvest(doc, (200, 10, 30))
    members["scoring/harvested_dinov2-base.json"] = scoring.dump_keyed(
        {"harvested/000.png": [0.98, 0.0, 0.1], "harvested/001.png": [0.0, 0.0, 1.0]}
    )
    # A cap of one, so exactly one of the two has to go.
    monkeypatch.setattr(encode, "MAX_HARVESTED", 1)

    removed = encode.prune_harvested(doc)

    assert removed == ["harvested/000.png"], "the near-duplicate should have gone, not the new angle"
    assert [r["path"] for r in encode.harvested(manifest)] == ["harvested/001.png"]
    assert len(encode.originals(manifest)) == 2, "an original was pruned"
