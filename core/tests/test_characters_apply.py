"""How a character reaches a render: by its references, or by a trained adapter, never both."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)


def _image(path: Path) -> Path:
    Image.new("RGB", (512, 512), (180, 150, 140)).save(path)
    return path


def test_a_trained_adapter_is_used_instead_of_the_references(tmp_path: Path) -> None:
    """Training produced an adapter nothing loaded, which made the Train button a lie. A character
    applies one way or the other: loading both would bind the identity twice."""
    from inline_core.characters import apply as ax
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library

    doc = encode.char_encode([_image(tmp_path / "ref.png")], name="Ada", description="green jacket")
    path = library.save(doc)
    arch = encode.FLUX2_KLEIN_ARCH

    untrained = ax.char_apply("Ada.char")
    assert untrained is not None and untrained.refs, "references apply when nothing is trained"
    assert untrained.lora is None

    doc = cf.read(path)
    encode.set_lora_payload(
        doc.manifest, doc.members, b"adapter-bytes",
        base="flux-2-klein-base-4b.safetensors", rank=16, steps=600, resolution=512,
    )
    cf.write(path, doc)

    trained = ax.char_apply("Ada.char")
    assert trained is not None
    assert trained.lora is not None, "a trained adapter must reach the runner"
    assert not trained.refs, "references step aside so the identity is not applied twice"
    # The adapter was bound to the description, so the prompt still has to carry it.
    assert "green jacket" in trained.prompt_prefix(1)

    doc = cf.read(path)
    doc.manifest.apply[arch] = "reference"
    cf.write(path, doc)
    chosen = ax.char_apply("Ada.char")
    assert chosen is not None
    assert chosen.lora is None and chosen.refs, "an explicit choice beats the default"


def test_the_chosen_mode_survives_a_round_trip(tmp_path: Path) -> None:
    """It lives in the .char, so exporting a character carries how it is meant to be applied."""
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library

    doc = encode.char_encode([_image(tmp_path / "ref.png")], name="Ada")
    path = library.save(doc)

    doc = cf.read(path)
    doc.manifest.apply["krea2"] = "lora"
    cf.write(path, doc)

    assert cf.read(path).manifest.apply == {"krea2": "lora"}


def test_krea2_applies_a_character_only_as_its_adapter(tmp_path: Path) -> None:
    """Krea 2 has no reference channel, so references cannot apply there at all. Before this, its
    runner never called char_apply and a trained Krea 2 adapter had nothing that loaded it."""
    from inline_core.characters import apply as ax
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library

    doc = encode.char_encode([_image(tmp_path / "ref.png")], name="Ada", description="green jacket")
    path = library.save(doc)

    # No Krea 2 adapter yet: nothing to apply, and no references are offered in its place.
    untrained = ax.char_apply("Ada.char", "krea2")
    assert untrained is not None
    assert untrained.lora is None and not untrained.refs

    doc = cf.read(path)
    encode.set_lora_payload(
        doc.manifest, doc.members, b"adapter-bytes", arch="krea2",
        base="krea2_turbo_bf16.safetensors", rank=16, steps=600, resolution=512,
    )
    cf.write(path, doc)

    trained = ax.char_apply("Ada.char", "krea2")
    assert trained is not None
    assert trained.lora is not None, "the Krea 2 adapter must reach the runner"
    assert not trained.refs
    assert "green jacket" in trained.prompt_prefix(1)

    # The Flux payload is untouched by any of this: the two archs are keyed separately.
    assert ax.char_apply("Ada.char").lora is None


def test_a_render_can_send_fewer_of_a_characters_references_than_it_holds(tmp_path: Path) -> None:
    """References are ~99.8% of what H3's conditioner reads and sit on the video's own rotary
    clock, so their count is the lever on how hard they pull. Dialling it must not mean
    re-encoding the character."""
    from inline_core.characters import apply as ax
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library

    roles = [cf.ROLE_FACE, cf.ROLE_FACE, cf.ROLE_BODY, cf.ROLE_CLOTH, cf.ROLE_CLOTH]
    images = [_image(tmp_path / f"r{i}.png") for i in range(len(roles))]
    library.save(encode.char_encode(images, name="Ada", roles=roles))
    arch = encode.FLUX2_KLEIN_ARCH

    assert len(ax.char_apply("Ada.char", arch, prefer="reference", limit=9).refs) == 5
    assert len(ax.char_apply("Ada.char", arch, prefer="reference", limit=3).refs) == 3
    face_only = ax.char_apply(
        "Ada.char", arch, prefer="reference", limit=9, keep_roles=(cf.ROLE_FACE,)
    )
    assert face_only is not None
    assert face_only.roles == [cf.ROLE_FACE] * 2, "body and clothing must not be sent"


def test_seedance_gets_at_image_tokens_not_prose(tmp_path: Path) -> None:
    """Seedance addresses a reference as `@ImageN`; prose names a position it cannot resolve, and
    H3's `<Picture N>` is another model's reserved label."""
    from inline_core.characters import apply as ax

    applied = ax.AppliedCharacter("Emmy", [object(), object()], "black hair")  # type: ignore[list-item]

    at_image = applied.prompt_prefix(3, style="at-image")
    assert at_image.startswith("@Image3 and @Image4 show Emmy,")
    assert "<Picture" not in at_image
    # The description still lands, and still ends as a sentence rather than running into the prompt.
    assert at_image.endswith("black hair. ")

    # One reference is singular, and must not emit a dangling "and".
    single = ax.AppliedCharacter("Emmy", [object()], "")  # type: ignore[list-item]
    assert single.prompt_prefix(1, style="at-image").startswith("@Image1 shows Emmy,")

    # The other styles are untouched by the new branch.
    assert applied.prompt_prefix(3).startswith("Images 3 and 4 show Emmy,")
    assert applied.prompt_prefix(3, style="token").startswith("<Picture 3> <Picture 4> show")


def test_a_role_line_never_re_declares_an_at_image_token(tmp_path: Path) -> None:
    """Repeating a reserved label made H3 replay its references as the opening frames. Nothing has
    shown Seedance is different, so a role line refers back in prose."""
    from inline_core.characters import apply as ax
    from inline_core.characters import charfile as cf

    applied = ax.AppliedCharacter(
        "Emmy", [object(), object()], "", roles=[cf.ROLE_FACE, cf.ROLE_BODY]  # type: ignore[list-item]
    )
    text = applied.prompt_prefix(1, style="at-image", role_lines=True)
    # Declared once each in the leading clause; the role lines point back without re-declaring.
    assert text.count("@Image1") == 1 and text.count("@Image2") == 1
    assert "Image 1 shows Emmy's face." in text
    assert "Image 2 shows Emmy's full body and build." in text


def test_the_fal_policy_normalises_a_reference_into_one_megapixel() -> None:
    """Hosted endpoints declare no frame grid, so the cap that matters is the wire: a reference
    travels base64 three times before it reaches fal."""
    from inline_core.characters import encode

    assert encode.reference_policy(encode.FAL_REF_ARCH) == encode.FAL_REF_POLICY
    big = Image.new("RGB", (4000, 3000))
    out = encode.normalise_reference(big, encode.reference_policy(encode.FAL_REF_ARCH))
    assert out.width * out.height <= 1024 * 1024
    # Aspect is preserved; only the area shrinks.
    assert abs(out.width / out.height - 4 / 3) < 0.02


def test_select_names_individual_references_for_a_sweep(tmp_path: Path) -> None:
    """Leave-one-out needs "this set minus reference i". `keep_roles` cannot say that: it removes a
    whole role, so dropping one of two face references was not expressible."""
    from inline_core.characters import apply as ax
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library

    images = [_image(tmp_path / f"r{i}.png") for i in range(3)]
    library.save(
        encode.char_encode(
            images, name="Sel", roles=[cf.ROLE_FACE, cf.ROLE_BODY, cf.ROLE_CLOTH]
        )
    )
    arch = encode.FLUX2_KLEIN_ARCH

    full = ax.char_apply("Sel.char", arch, prefer="reference")
    assert full is not None and full.roles == [cf.ROLE_FACE, cf.ROLE_BODY, cf.ROLE_CLOTH]

    minus_body = ax.char_apply("Sel.char", arch, prefer="reference", select=[0, 2])
    assert minus_body is not None and minus_body.roles == [cf.ROLE_FACE, cf.ROLE_CLOTH]

    # Order follows the payload, not the order the caller happened to write, so the prompt numbers
    # a sweep produces match the numbers a normal render would.
    assert [r.path for r in minus_body.refs] == [full.refs[0].path, full.refs[2].path]
    reversed_ask = ax.char_apply("Sel.char", arch, prefer="reference", select=[2, 0])
    assert reversed_ask is not None
    assert [r.path for r in reversed_ask.refs] == [r.path for r in minus_body.refs]


def test_select_tolerates_a_stale_index_rather_than_failing_a_sweep(tmp_path: Path) -> None:
    """A long sweep must not die on iteration 40 because a duplicate or an out-of-range index crept
    into a combination."""
    from inline_core.characters import apply as ax
    from inline_core.characters import encode, library

    library.save(
        encode.char_encode([_image(tmp_path / f"s{i}.png") for i in range(2)], name="Stale")
    )
    applied = ax.char_apply(
        "Stale.char", encode.FLUX2_KLEIN_ARCH, prefer="reference", select=[1, 1, 99]
    )
    assert applied is not None and len(applied.refs) == 1


def test_select_runs_before_the_cap_so_the_cap_still_divides_by_role(tmp_path: Path) -> None:
    """`_fit_roles` splits the slots by role. Selecting after it would hand the cap a set it had
    already balanced, and the role ratio would be computed against references that are not sent."""
    from inline_core.characters import apply as ax
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library

    images = [_image(tmp_path / f"c{i}.png") for i in range(4)]
    library.save(
        encode.char_encode(
            images,
            name="Cap",
            roles=[cf.ROLE_FACE, cf.ROLE_FACE, cf.ROLE_BODY, cf.ROLE_CLOTH],
        )
    )
    applied = ax.char_apply(
        "Cap.char", encode.FLUX2_KLEIN_ARCH, prefer="reference", select=[0, 2, 3], limit=2
    )
    assert applied is not None and len(applied.refs) == 2
    # Face survives the cut: it is what identity is carried by, and it was in the selection.
    assert cf.ROLE_FACE in applied.roles
