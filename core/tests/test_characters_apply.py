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
