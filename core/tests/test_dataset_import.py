"""Reading a published dataset folder: captions and pairing, in the shapes that actually ship."""

from __future__ import annotations

import json
from pathlib import Path

from inline_core.studio.dataset_import import read_metadata
from inline_core.training.dataset import is_reference_name, reference_for_name


def test_lightricks_dataset_json_pairs_by_reference_path(tmp_path: Path) -> None:
    """Their IC-LoRA sets ship one JSON array with explicit target/reference paths."""
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            [
                {"caption": "a bear", "media_path": "bear.mp4",
                 "reference_path": "bear_reference.mp4"},
                {"caption": "a swan", "media_path": "swan.mp4",
                 "reference_path": "swan_reference.mp4"},
            ]
        )
    )
    entries = read_metadata(tmp_path)
    assert set(entries) == {"bear.mp4", "swan.mp4"}
    assert entries["bear.mp4"].reference == "bear_reference.mp4"
    assert entries["bear.mp4"].caption == "a bear"


def test_hugging_face_metadata_jsonl_finds_a_prefixed_caption_key(tmp_path: Path) -> None:
    """One object per line, `file_name`, and a caption key nobody would guess.

    The pixel-art set names it `caption-nvila15b`, so the key is discovered rather than listed.
    """
    (tmp_path / "metadata.jsonl").write_text(
        '{"file_name": "0001.mp4", "caption-nvila15b": "a pixel scene"}\n'
        '{"file_name": "0002.mp4", "caption-nvila15b": "another"}\n'
    )
    entries = read_metadata(tmp_path)
    assert entries["0001.mp4"].caption == "a pixel scene"
    assert entries["0001.mp4"].reference is None


def test_a_malformed_sidecar_falls_back_rather_than_failing_the_import(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text("{not json")
    assert read_metadata(tmp_path) == {}


def test_no_sidecar_is_not_an_error(tmp_path: Path) -> None:
    assert read_metadata(tmp_path) == {}


def test_pairing_falls_back_to_filenames_with_no_sidecar() -> None:
    """`bear.mp4` + `bear_reference.mp4` pairs with no metadata at all, and the reference is not
    itself a training item."""
    names = {"bear.mp4", "bear_reference.mp4", "solo.mp4"}
    assert reference_for_name(Path("bear.mp4"), names) == "bear_reference.mp4"
    assert reference_for_name(Path("solo.mp4"), names) is None
    assert is_reference_name(Path("bear_reference.mp4"))
    assert not is_reference_name(Path("bear.mp4"))


def test_our_own_export_spelling_pairs_too() -> None:
    names = {"0000.mp4", "0000.ref.mp4"}
    assert reference_for_name(Path("0000.mp4"), names) == "0000.ref.mp4"
    assert is_reference_name(Path("0000.ref.mp4"))


def test_every_registered_training_channel_has_a_method() -> None:
    """The handler table names methods by string, so a refactor can delete one silently.

    That is exactly what happened: a splice that rewrote `add_from_path` also removed
    `inspect_dataset_repo`, and nothing failed until a user clicked Check. Types cannot see through
    the `lambda` in the registration table, so this walks it instead.
    """
    import inspect as _inspect
    import re

    from inline_core.studio import handlers
    from inline_core.studio.training import Training

    source = _inspect.getsource(handlers)
    called = set(re.findall(r"training\.([a-z_]+)\(", source))
    assert called, "no training methods found in the handler table"
    missing = sorted(name for name in called if not hasattr(Training, name))
    assert not missing, f"handlers reference methods Training does not have: {missing}"
