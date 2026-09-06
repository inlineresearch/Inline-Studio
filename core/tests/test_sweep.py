"""Leave-one-out arithmetic: the paired difference, and what it protects against."""

from __future__ import annotations

import pytest

from inline_core.characters import sweep


def test_leave_one_out_is_the_full_set_plus_one_drop_each() -> None:
    assert sweep.leave_one_out(3) == ["full", "drop_0", "drop_1", "drop_2"]
    assert sweep.selection_for("full", 3) == [0, 1, 2]
    assert sweep.selection_for("drop_1", 3) == [0, 2]


def test_a_sweep_refuses_below_two_references() -> None:
    """Dropping the only reference leaves nothing to render the character from."""
    with pytest.raises(ValueError, match="at least 2"):
        sweep.leave_one_out(1)


def _cell(combo: str, prompt: str, seed: int, score: float) -> sweep.Cell:
    return sweep.Cell(combination=combo, prompt=prompt, seed=seed, score=score)


def test_the_paired_difference_survives_a_framing_gap_a_mean_does_not() -> None:
    """The confound this whole design exists for. Framing moves the score far more than a
    reference does: the same character measured 88.4 close-up and 49.7 wide. Reference 0 is worth a
    real +8 here, and reference 1 nothing, but the close-up prompt sits ~40 points above the wide
    one. Averaging across prompts buries the effect in that gap; differencing inside each cell
    recovers it exactly."""
    cells = [
        # close-up prompt, ~88 band
        _cell("full", "closeup", 1, 88.0),
        _cell("drop_0", "closeup", 1, 80.0),
        _cell("drop_1", "closeup", 1, 88.0),
        # wide prompt, ~49 band
        _cell("full", "wide", 1, 49.0),
        _cell("drop_0", "wide", 1, 41.0),
        _cell("drop_1", "wide", 1, 49.0),
    ]
    got = {c.index: c for c in sweep.contributions(cells, 2)}

    assert got[0].delta == 8.0, "removing reference 0 costs 8 points, in both framings"
    assert got[0].spread == 0.0
    assert got[1].delta == 0.0, "reference 1 earns nothing"
    assert got[0].verdict == "keep"
    assert got[1].verdict == "neutral"

    # The trap, stated as a number: drop_0's own mean (60.5) beats full's mean on the wide prompt
    # (49.0), so ranking combinations on a cross-prompt mean would rate a worse set above a better
    # one purely on which prompts it drew.
    assert (80.0 + 41.0) / 2 > 49.0


def test_per_prompt_keeps_a_full_body_loss_visible() -> None:
    """A reference can help a close-up and hurt a wide shot. One averaged number hides that; the
    per-prompt breakdown is what stops a wardrobe reference being kept for the wrong reason."""
    cells = [
        _cell("full", "closeup", 1, 90.0),
        _cell("drop_0", "closeup", 1, 80.0),
        _cell("full", "wide", 1, 50.0),
        _cell("drop_0", "wide", 1, 56.0),
    ]
    got = sweep.contributions(cells, 1)[0]
    assert got.delta == 2.0, "the mean alone reads as a mild keep"
    assert got.per_prompt == {"closeup": 10.0, "wide": -6.0}


def test_a_column_missing_one_render_is_dropped_from_every_combination() -> None:
    """Otherwise a combination that failed on the hard prompt is compared on the easy ones only,
    and looks better than the one that ran everywhere."""
    cells = [
        _cell("full", "closeup", 1, 90.0),
        _cell("drop_0", "closeup", 1, 80.0),
        _cell("full", "wide", 1, 50.0),
        sweep.Cell(combination="drop_0", prompt="wide", seed=1, score=None),
    ]
    got = sweep.contributions(cells, 1)[0]
    assert got.pairs == 1, "only the complete column counts"
    assert got.delta == 10.0
    assert [c.prompt for c in sweep.balanced_cells(cells)] == ["closeup", "closeup"]


def test_seeds_are_paired_too() -> None:
    """A repeat is a different seed, so the difference is still taken against the same draw."""
    cells = [
        _cell("full", "p", 1, 70.0),
        _cell("drop_0", "p", 1, 60.0),
        _cell("full", "p", 2, 90.0),
        _cell("drop_0", "p", 2, 84.0),
    ]
    got = sweep.contributions(cells, 1)[0]
    assert got.pairs == 2
    assert got.delta == 8.0, "(10 + 6) / 2, never (80 - 72)"
    assert got.spread == 2.0


def test_nothing_measurable_yields_no_contribution_rather_than_a_zero() -> None:
    cells = [sweep.Cell(combination="full", prompt="p", seed=1, score=None)]
    assert sweep.contributions(cells, 1) == []


def test_a_delta_carries_how_far_the_composition_moved_with_it() -> None:
    """Measured on a real sweep: inside one prompt at one seed, face-area share correlated +0.79
    with the score. Pairing removes the prompt and the seed, but the reference set still changes
    how the model frames the shot, and the score follows that. So a delta reports the shift beside
    it rather than passing composition off as identity."""
    cells = [
        sweep.Cell("full", "p", 1, score=60.0, face_fraction=0.132),
        # Removing this reference turned the head: a smaller face, and a lower score with it.
        sweep.Cell("drop_0", "p", 1, score=45.0, face_fraction=0.111),
        # This one moved the score without moving the framing, which is the trustworthy shape.
        sweep.Cell("drop_1", "p", 1, score=52.0, face_fraction=0.131),
    ]
    got = {c.index: c for c in sweep.contributions(cells, 2)}

    assert got[0].delta == 15.0 and got[0].framing_shift == pytest.approx(2.1, abs=0.01)
    assert got[1].delta == 8.0 and got[1].framing_shift == pytest.approx(0.1, abs=0.01)


def test_framing_shift_is_zero_when_nothing_measured_it() -> None:
    cells = [
        sweep.Cell("full", "p", 1, score=60.0),
        sweep.Cell("drop_0", "p", 1, score=50.0),
    ]
    assert sweep.contributions(cells, 1)[0].framing_shift == 0.0


def _cells(combo: str, scores: dict[tuple[str, int], float]) -> list[sweep.Cell]:
    return [
        sweep.Cell(combination=combo, prompt=prompt, seed=seed, score=score)
        for (prompt, seed), score in scores.items()
    ]


def test_a_retest_uses_only_the_fresh_cells_and_pools_over_both() -> None:
    main = _cells("full", {("p", 1): 70.0, ("p", 2): 72.0}) + _cells(
        "drop_1", {("p", 1): 60.0, ("p", 2): 62.0}
    )
    retest = _cells("full", {("p", 9): 70.0}) + _cells("drop_1", {("p", 9): 66.0})

    found = sweep.confirm(1, initial=10.0, main=main, retest=retest)

    assert found.retest == 4.0
    assert found.retest_pairs == 1
    # Pooled over all three pairs: (10 + 10 + 4) / 3.
    assert found.pooled == 8.0
    assert found.pooled_pairs == 3


def test_a_retest_that_lands_inside_the_band_does_not_confirm_the_flag() -> None:
    main = _cells("full", {("p", 1): 60.0}) + _cells("drop_0", {("p", 1): 65.0})
    retest = _cells("full", {("p", 9): 60.0}) + _cells("drop_0", {("p", 9): 60.5})

    found = sweep.confirm(0, initial=-5.0, main=main, retest=retest)

    assert found.verdict == "not confirmed"
    assert found.agrees is True


def test_a_retest_on_the_other_side_of_zero_reverses_the_flag() -> None:
    retest = _cells("full", {("p", 9): 70.0}) + _cells("drop_0", {("p", 9): 60.0})

    found = sweep.confirm(0, initial=-5.0, main=[], retest=retest)

    assert found.verdict == "reversed"
    assert found.agrees is False


def test_only_the_flagged_references_are_worth_re_testing() -> None:
    found = [
        sweep.Contribution(index=0, delta=9.0, spread=1.0, pairs=4, per_prompt={}),
        sweep.Contribution(index=1, delta=-6.0, spread=1.0, pairs=4, per_prompt={}),
        sweep.Contribution(index=2, delta=0.4, spread=1.0, pairs=4, per_prompt={}),
    ]

    assert [c.index for c in sweep.flagged(found)] == [1]


def test_confirmation_cells_never_unbalance_the_sweep() -> None:
    """The extra columns hold two combinations, so folding them in would break every comparison."""
    main = (
        _cells("full", {("p", 1): 70.0})
        + _cells("drop_0", {("p", 1): 60.0})
        + _cells("drop_1", {("p", 1): 65.0})
    )
    retest = _cells("full", {("p", 9): 70.0}) + _cells("drop_0", {("p", 9): 55.0})

    assert [c.delta for c in sweep.contributions(main, 2)] == [10.0, 5.0]
    assert [c.delta for c in sweep.contributions([*main, *retest], 2)] == [10.0, 5.0]
