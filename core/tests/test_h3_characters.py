"""Characters on MiniMax H3: the reference policy, the prompt form, and scoring a video take."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inline_core.characters import encode
from inline_core.characters.apply import AppliedCharacter

# --- the reference policy -------------------------------------------------------------------------


def _resize(size: tuple[int, int], arch: str) -> tuple[int, int]:
    pytest.importorskip("PIL")
    from PIL import Image

    out = encode.normalise_reference(Image.new("RGB", size), encode.reference_policy(arch))
    return out.width, out.height


def test_h3_is_offered_a_reference_payload_at_all() -> None:
    """One entry in this table was what forced every model but FLUX.2 down the adapter route."""
    assert encode.MINIMAX_H3_ARCH in encode.REFERENCE_POLICIES


def test_h3_scales_onto_a_short_edge_rather_than_under_an_area_cap() -> None:
    """FLUX.2 caps area and only ever shrinks; H3 scales the short side onto 2048 either way."""
    assert _resize((3000, 2000), encode.MINIMAX_H3_ARCH) == (3072, 2048)
    assert _resize((640, 640), encode.MINIMAX_H3_ARCH) == (2048, 2048)
    assert _resize((3000, 2000), encode.FLUX2_KLEIN_ARCH) == (1248, 832)


def test_h3_has_no_area_cap() -> None:
    """The vendored packer says a 4:1 reference is 8192x2048, so the payload must agree."""
    assert _resize((8000, 2000), encode.MINIMAX_H3_ARCH) == (8192, 2048)


def test_an_out_of_range_aspect_is_refused_while_compiling() -> None:
    """The vendored blocks raise on this mid-run; catching it here names the reference instead."""
    pytest.importorskip("PIL")
    from PIL import Image

    with pytest.raises(ValueError, match="within 1:4"):
        encode.normalise_reference(Image.new("RGB", (5000, 900)), encode.MINIMAX_H3_POLICY)


def test_both_policies_land_on_their_own_grid() -> None:
    for arch, grid in ((encode.FLUX2_KLEIN_ARCH, 16), (encode.MINIMAX_H3_ARCH, 32)):
        width, height = _resize((1234, 987), arch)
        assert width % grid == 0 and height % grid == 0


# --- the prompt form ------------------------------------------------------------------------------


def _character(refs: int) -> AppliedCharacter:
    return AppliedCharacter("Ada", ["ref"] * refs, "freckles, dark hair")


def test_h3_names_references_as_tokens_not_as_ordinal_prose() -> None:
    """H3 resolves `<Picture N>`; FLUX.2's "Images 1 and 2 show" names positions H3 cannot see."""
    prefix = _character(2).prompt_prefix(3, style="token")
    assert prefix.startswith("<Picture 3> <Picture 4> show Ada")
    assert "Images" not in prefix


def test_the_offset_is_where_the_character_lands_not_where_it_starts() -> None:
    """The character's images are appended after whatever the user wired, so the numbering has to
    continue rather than restart."""
    assert _character(1).prompt_prefix(5, style="token").startswith("<Picture 5> shows Ada")


def test_flux2_keeps_its_prose() -> None:
    assert _character(2).prompt_prefix(1).startswith("Images 1 and 2 show Ada")


def test_an_adapter_only_character_is_the_description_alone() -> None:
    """No references to name, so there is nothing positional to say in either style."""
    assert AppliedCharacter("Ada", [], "freckles").prompt_prefix(1, style="token") == "freckles "


# --- scoring a video take -------------------------------------------------------------------------


def _scored(value: float, subject: bool = True) -> dict[str, Any]:
    return {"score": value, "subjectCounted": subject}


def test_a_video_score_is_the_median_of_the_frames_that_measured(monkeypatch) -> None:
    """A mean lets one blurred frame drag the number; a median survives one bad and one lucky."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: ["f"] * 5)
    scores = iter([_scored(80), _scored(12), _scored(78), _scored(82), _scored(79)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["score"] == 79.0
    assert out["frames"] == 5


def test_a_frame_with_no_face_drops_out_rather_than_scoring_zero(monkeypatch) -> None:
    """`score` returns None for unmeasurable, which is not the same claim as a score of nothing."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: ["f"] * 3)
    scores = iter([_scored(90), None, _scored(70)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["frames"] == 2, "the unmeasurable frame is not counted"
    assert out["score"] == 80.0


def test_one_face_only_frame_makes_the_whole_score_face_only(monkeypatch) -> None:
    """Reporting a blended number when a frame's subject term was noise hides the dropped term."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: ["f"] * 2)
    scores = iter([_scored(80), _scored(70, subject=False)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["subjectCounted"] is False


def test_a_clip_that_cannot_be_read_scores_nothing(monkeypatch) -> None:
    """A missing ffmpeg means no score, never a failed render."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: [])
    assert mod._score_video(Path("clip.mp4"), {}, [], [], []) is None
