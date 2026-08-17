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
