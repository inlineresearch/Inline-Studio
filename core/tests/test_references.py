"""Reference ordering and limits. Order is what the prompt addresses, so it is the thing to pin."""

from __future__ import annotations

import pytest

from inline_core.errors import ComponentError
from inline_core.models.references import (
    ReferenceKind,
    ReferenceLimits,
    check_limits,
    collect_references,
    count_by_kind,
    describe,
)

#: MiniMax H3's stated limits.
H3 = ReferenceLimits(max_images=9, max_videos=3, max_audio=3, max_total=12)


def test_wiring_order_within_a_port_is_preserved() -> None:
    refs = collect_references({"references": ["lead", "dog", "car"]})
    assert [r.value for r in refs] == ["lead", "dog", "car"]
    assert [r.index for r in refs] == [1, 2, 3]


def test_each_kind_is_numbered_from_one_independently() -> None:
    refs = collect_references(
        {"references": ["a", "b"], "video": ["clip"], "audio": ["voice", "room"]}
    )
    assert describe(refs) == "<Picture 1>, <Picture 2>, <Video 1>, <Audio 1>, <Audio 2>"


def test_ports_are_laid_out_images_then_video_then_audio() -> None:
    refs = collect_references({"audio": ["v"], "video": ["c"], "references": ["i"]})
    assert [r.kind for r in refs] == [
        ReferenceKind.IMAGE,
        ReferenceKind.VIDEO,
        ReferenceKind.AUDIO,
    ]


def test_unwired_slots_do_not_consume_a_number() -> None:
    """A gap would shift every later index, and the prompt addresses them by index."""
    refs = collect_references({"references": ["a", None, "b"]})
    assert [(r.value, r.index) for r in refs] == [("a", 1), ("b", 2)]


def test_no_wired_references_is_an_empty_list_not_an_error() -> None:
    assert collect_references({}) == ()
    assert describe(()) == "none"


def test_counts_cover_every_kind_even_at_zero() -> None:
    counts = count_by_kind(collect_references({"references": ["a"]}))
    assert counts == {ReferenceKind.IMAGE: 1, ReferenceKind.VIDEO: 0, ReferenceKind.AUDIO: 0}


def test_too_many_of_one_kind_says_how_many_to_unwire() -> None:
    refs = collect_references({"references": [str(i) for i in range(11)]})
    with pytest.raises(ComponentError, match="at most 9. Unwire 2"):
        check_limits(refs, H3)


def test_the_total_limit_bites_before_any_single_kind_does() -> None:
    refs = collect_references(
        {"references": [str(i) for i in range(9)], "video": ["a", "b", "c"], "audio": ["v"]}
    )
    assert count_by_kind(refs)[ReferenceKind.IMAGE] == 9  # each kind is within its own limit
    with pytest.raises(ComponentError, match="at most 12 in total. Unwire 1"):
        check_limits(refs, H3)


def test_limits_can_be_enforced_during_collection() -> None:
    with pytest.raises(ComponentError, match="video references"):
        collect_references({"video": ["a", "b", "c", "d"]}, limits=H3)


def test_a_full_but_legal_request_passes() -> None:
    refs = collect_references(
        {"references": [str(i) for i in range(6)], "video": ["a", "b", "c"], "audio": ["x", "y"]},
        limits=H3,
    )
    assert len(refs) == 11
