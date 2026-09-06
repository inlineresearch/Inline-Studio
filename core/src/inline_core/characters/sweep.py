"""Leave-one-out reference sweeps: which references a character actually needs.

The arithmetic here is the whole feature, so it is kept apart from the service that renders and
scores. Two rules it exists to enforce:

Score a combination only against the *same* cell, never against a mean. Framing dominates this
score - the same references measured 88.4 on a close-up and 49.7 on a wide shot - so a combination
ranked on an average over a varying prompt mix is ranked on its prompt mix.

Leave one out rather than search subsets. A reference's worth is then one difference per cell,
which is what the verdict wants, and every reference is measured over the same cells rather than
over whichever combinations a search happened to explore.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

#: The combination that keeps every reference. Every other combination is measured against it.
FULL = "full"

#: Below two references there is nothing to leave out: dropping the only face leaves no identity.
MIN_REFS = 2


def combination_key(dropped: int | None) -> str:
    return FULL if dropped is None else f"drop_{dropped}"


def leave_one_out(count: int) -> list[str]:
    """The full set, then the same set missing each reference in turn."""
    if count < MIN_REFS:
        raise ValueError(
            f"A sweep needs at least {MIN_REFS} references to leave one out; this character has "
            f"{count}."
        )
    return [FULL, *(combination_key(i) for i in range(count))]


def selection_for(key: str, count: int) -> list[int]:
    """Which reference positions a combination sends."""
    if key == FULL:
        return list(range(count))
    dropped = int(key.removeprefix("drop_"))
    return [i for i in range(count) if i != dropped]


@dataclass
class Cell:
    """One render: a combination on one prompt at one seed, and what it scored."""

    combination: str
    prompt: str
    seed: int
    score: float | None = None
    path: str = ""
    #: Face area as a share of the frame. The score tracks this hard, so a delta measured across a
    #: composition shift is not an identity claim - see `Contribution.framing_shift`.
    face_fraction: float | None = None
    #: How well the clothes matched, when the garment was in frame. None where it was not.
    wardrobe: float | None = None
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class Contribution:
    """What one reference was worth, measured only where both sides of the pair exist."""

    index: int
    delta: float
    spread: float
    pairs: int
    per_prompt: dict[str, float]
    #: Mean change in face-area share, in percentage points, between the paired renders. Measured
    #: at 0.79 correlation with the score inside one prompt, so a delta that comes with a large
    #: shift is reporting composition as much as identity.
    framing_shift: float = 0.0
    #: The same paired difference measured on the clothes. A wardrobe reference cannot earn its
    #: place on a face score, so without this the sweep recommended deleting every one of them.
    wardrobe_delta: float | None = None
    wardrobe_pairs: int = 0

    @property
    def verdict(self) -> str:
        """Four coarse bands. A number this noisy cannot carry a finer claim than keep or look."""
        if self.delta >= _KEEP_DELTA:
            return "keep"
        if self.delta <= _DROP_DELTA:
            # It costs identity and pays for it in clothes, which is what a wardrobe reference is
            # for. Removing it is a different decision from removing one that pays nothing.
            if self.wardrobe_delta is not None and self.wardrobe_delta >= _KEEP_DELTA:
                return "keeps-the-wardrobe"
            return "consider-removing"
        return "neutral"


#: Uncalibrated, and deliberately wide. Removing a reference that carries identity should cost far
#: more than this; anything inside the band is noise until someone measures where the band belongs.
_KEEP_DELTA = 2.0
_DROP_DELTA = -2.0


def balanced_cells(cells: list[Cell]) -> list[Cell]:
    """Only the (prompt, seed) columns where every combination scored.

    A column missing one render would otherwise compare a combination that ran on the easy prompts
    against one that ran on all of them.
    """
    combos = {c.combination for c in cells}
    scored: dict[tuple[str, int], set[str]] = {}
    for cell in cells:
        if cell.score is not None:
            scored.setdefault((cell.prompt, cell.seed), set()).add(cell.combination)
    complete = {column for column, seen in scored.items() if seen >= combos}
    return [c for c in cells if (c.prompt, c.seed) in complete and c.score is not None]


def contributions(cells: list[Cell], count: int) -> list[Contribution]:
    """Each reference's worth: `score(full) - score(without it)`, differenced inside the cell."""
    balanced = balanced_cells(cells)
    by_key: dict[tuple[str, str, int], float] = {
        (c.combination, c.prompt, c.seed): float(c.score) for c in balanced if c.score is not None
    }
    by_face: dict[tuple[str, str, int], float] = {
        (c.combination, c.prompt, c.seed): float(c.face_fraction)
        for c in balanced
        if c.face_fraction is not None
    }
    by_cloth: dict[tuple[str, str, int], float] = {
        (c.combination, c.prompt, c.seed): float(c.wardrobe)
        for c in balanced
        if c.wardrobe is not None
    }
    out: list[Contribution] = []
    for index in range(count):
        key = combination_key(index)
        deltas: list[float] = []
        shifts: list[float] = []
        dressed: list[float] = []
        per_prompt: dict[str, list[float]] = {}
        for (combo, prompt, seed), value in by_key.items():
            if combo != key:
                continue
            whole = by_key.get((FULL, prompt, seed))
            if whole is None:
                continue
            # Differenced within the cell, averaged only afterwards: the prompt and the seed are
            # identical on both sides, so what is left is the reference.
            delta = whole - value
            deltas.append(delta)
            per_prompt.setdefault(prompt, []).append(delta)
            face_whole = by_face.get((FULL, prompt, seed))
            face_part = by_face.get((combo, prompt, seed))
            if face_whole is not None and face_part is not None:
                shifts.append((face_whole - face_part) * 100)
            cloth_whole = by_cloth.get((FULL, prompt, seed))
            cloth_part = by_cloth.get((combo, prompt, seed))
            if cloth_whole is not None and cloth_part is not None:
                dressed.append(cloth_whole - cloth_part)
        if not deltas:
            continue
        out.append(
            Contribution(
                index=index,
                delta=round(statistics.mean(deltas), 1),
                spread=round(statistics.pstdev(deltas), 1) if len(deltas) > 1 else 0.0,
                pairs=len(deltas),
                per_prompt={p: round(statistics.mean(v), 1) for p, v in per_prompt.items()},
                framing_shift=round(statistics.mean(shifts), 2) if shifts else 0.0,
                wardrobe_delta=round(statistics.mean(dressed), 1) if dressed else None,
                wardrobe_pairs=len(dressed),
            )
        )
    return out


@dataclass
class CombinationScore:
    """One reference set's standing, ranked on the low end of its spread, never its best cell."""

    key: str
    mean: float
    #: Mean minus one spread: a set tested twice and lucky should not outrank one tested ten times.
    adjusted: float
    spread: float
    tested: int


def combination_scores(cells: list[Cell]) -> list[CombinationScore]:
    """Every combination ranked confidence-adjusted. Ranking on a raw max is what this avoids."""
    balanced = balanced_cells(cells)
    grouped: dict[str, list[float]] = {}
    for cell in balanced:
        if cell.score is not None:
            grouped.setdefault(cell.combination, []).append(float(cell.score))
    out = [
        CombinationScore(
            key=key,
            mean=round(statistics.mean(values), 1),
            adjusted=round(
                statistics.mean(values) - (statistics.pstdev(values) if len(values) > 1 else 0.0),
                1,
            ),
            spread=round(statistics.pstdev(values), 1) if len(values) > 1 else 0.0,
            tested=len(values),
        )
        for key, values in grouped.items()
    ]
    return sorted(out, key=lambda c: c.adjusted, reverse=True)


def plateaued(cells: list[Cell], window: int, delta: float) -> bool:
    """Whether the best adjusted score has stopped moving over the last `window` columns."""
    columns = sorted({(c.prompt, c.seed) for c in cells if c.score is not None})
    if len(columns) <= window:
        return False
    best_now = combination_scores(cells)
    earlier = [c for c in cells if (c.prompt, c.seed) in set(columns[:-window])]
    best_then = combination_scores(earlier)
    if not best_now or not best_then:
        return False
    return abs(best_now[0].adjusted - best_then[0].adjusted) < delta


@dataclass
class Confirmation:
    """A flagged reference re-measured on seeds the sweep never used.

    The sweep's own verdict comes from the cells that produced it, so a reference flagged by an
    unlucky pair stays flagged. Re-testing on fresh seeds is an independent measurement of the same
    quantity, and `pooled` is the number to act on because it rests on every pair taken together.
    """

    index: int
    #: The sweep's delta, kept so the report can show whether the re-test agreed with it.
    initial: float
    retest: float
    retest_spread: float
    retest_pairs: int
    pooled: float
    pooled_pairs: int

    @property
    def agrees(self) -> bool:
        return self.retest_pairs > 0 and (self.retest <= 0) == (self.initial <= 0)

    @property
    def verdict(self) -> str:
        if self.retest_pairs == 0:
            return "not re-tested"
        if self.retest <= _DROP_DELTA:
            return "removal confirmed"
        if self.retest >= _KEEP_DELTA:
            return "reversed"
        return "not confirmed"


def flagged(found: list[Contribution]) -> list[Contribution]:
    """The references the sweep would drop, which are the only ones worth spending a re-test on."""
    return [c for c in found if c.verdict == "consider-removing"]


def _paired(cells: list[Cell], index: int) -> Contribution | None:
    """One reference's contribution over whatever cells are given, measured the sweep's way."""
    pair = {FULL, combination_key(index)}
    found = contributions([c for c in cells if c.combination in pair], index + 1)
    return next((c for c in found if c.index == index), None)


def confirm(index: int, initial: float, main: list[Cell], retest: list[Cell]) -> Confirmation:
    fresh = _paired(retest, index)
    both = _paired([*main, *retest], index)
    return Confirmation(
        index=index,
        initial=initial,
        retest=fresh.delta if fresh else 0.0,
        retest_spread=fresh.spread if fresh else 0.0,
        retest_pairs=fresh.pairs if fresh else 0,
        pooled=both.delta if both else initial,
        pooled_pairs=both.pairs if both else 0,
    )
