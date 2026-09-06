"""The sweep loop: ordering, cancellation, preflight, and the events it streams.

The render is a seam, so the loop is testable without a GPU - which is the point: the loop's job is
ordering and bookkeeping, and that is what breaks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from inline_core.studio import finetune  # noqa: E402


class _Events:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def broadcast(self, channel: str, payload: dict) -> None:
        self.sent.append((channel, payload))

    def lines(self) -> list[str]:
        return [p["line"] for c, p in self.sent if c == finetune.LOG_EVENT]


def _character(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refs: int) -> str:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.chdir(tmp_path)
    from inline_core.characters import encode, library

    paths = []
    for i in range(refs):
        p = tmp_path / f"r{i}.png"
        Image.new("RGB", (64, 64), (10 * i, 90, 140)).save(p)
        paths.append(p)
    library.save(encode.char_encode(paths, name="Sweepy"))
    return "Sweepy.char"


async def _run(service: finetune.CharacterTuning, spec, out: Path) -> str:
    started = service.start(spec, out)
    for _ in range(400):
        await asyncio.sleep(0.01)
        if service.status(started["id"])["status"] in ("done", "cancelled", "error"):
            break
    return started["id"]


def test_a_sweep_renders_every_combination_on_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Balanced by construction: 3 refs is 4 combinations, and each must see the same 2 cells."""
    char = _character(tmp_path, monkeypatch, refs=3)
    seen: list[tuple[str, int]] = []

    def render(req: finetune.CellRequest) -> Path:
        seen.append((req.prompt, req.seed))
        req.out.write_bytes(b"x")
        return req.out

    events = _Events()
    service = finetune.CharacterTuning(events, lambda _t: render)
    monkeypatch.setattr(service, "_score", lambda path, chosen: {"score": 50.0})
    spec = finetune.SweepSpec(character=char, prompts=("a", "b"), seeds=(1,))

    run_id = asyncio.run(_run(service, spec, tmp_path / "out"))
    result = service.result(run_id)

    assert result["status"] == "done"
    assert result["refs"] == 3
    assert result["done"] == 4 * 2 * 1 == result["total"]
    # Each (prompt, seed) column is rendered by every combination before the next column starts.
    assert seen[:4] == [("a", 1)] * 4
    assert seen[4:] == [("b", 1)] * 4


def test_cancelling_leaves_a_balanced_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Combination-inner ordering exists for this: a stop mid-sweep still leaves whole columns, so
    the paired difference has both sides of every pair it reports."""
    char = _character(tmp_path, monkeypatch, refs=2)
    events = _Events()
    service: finetune.CharacterTuning

    def render(req: finetune.CellRequest) -> Path:
        req.out.write_bytes(b"x")
        # Cancel once the first column is complete: 2 refs is 3 combinations.
        if len(service._runs[next(iter(service._runs))].cells) >= 2:
            service.cancel(next(iter(service._runs)))
        return req.out

    service = finetune.CharacterTuning(events, lambda _t: render)
    monkeypatch.setattr(service, "_score", lambda path, chosen: {"score": 70.0})
    spec = finetune.SweepSpec(character=char, prompts=("a", "b", "c"), seeds=(1,))

    run_id = asyncio.run(_run(service, spec, tmp_path / "out"))
    result = service.result(run_id)
    assert result["status"] == "cancelled"
    assert result["done"] < result["total"], "it really did stop early"
    # Every contribution it reports is backed by complete pairs.
    for contribution in result["contributions"]:
        assert contribution["pairs"] >= 1


def test_a_failed_render_drops_its_column_rather_than_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    char = _character(tmp_path, monkeypatch, refs=2)

    def render(req: finetune.CellRequest) -> Path:
        if req.seed == 2 and req.select == (0,):
            raise RuntimeError("cuda hiccup")
        req.out.write_bytes(b"x")
        return req.out

    events = _Events()
    service = finetune.CharacterTuning(events, lambda _t: render)
    monkeypatch.setattr(service, "_score", lambda path, chosen: {"score": 60.0})
    spec = finetune.SweepSpec(character=char, prompts=("a",), seeds=(1, 2))

    run_id = asyncio.run(_run(service, spec, tmp_path / "out"))
    assert service.status(run_id)["status"] == "done", "one bad cell does not fail the sweep"
    assert any("failed: cuda hiccup" in line for line in events.lines())


def test_preflight_refuses_a_character_too_small_to_leave_one_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    char = _character(tmp_path, monkeypatch, refs=1)
    with pytest.raises(finetune.SweepRefused, match="at least 2"):
        finetune.preflight(finetune.SweepSpec(character=char, prompts=("a",), seeds=(1,)))


def test_preflight_refuses_a_character_that_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _character(tmp_path, monkeypatch, refs=2)
    with pytest.raises(finetune.SweepRefused, match="no longer in"):
        finetune.preflight(finetune.SweepSpec(character="Ghost.char", prompts=("a",), seeds=(1,)))


def test_a_large_sweep_asks_before_it_spends_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cell is a full render, so the product runs away quickly."""
    char = _character(tmp_path, monkeypatch, refs=5)
    service = finetune.CharacterTuning(_Events(), lambda _t: lambda req: req.out)
    spec = finetune.SweepSpec(
        character=char, prompts=tuple(f"p{i}" for i in range(10)), seeds=(1, 2)
    )

    async def go() -> None:
        with pytest.raises(finetune.SweepRefused, match="Confirm"):
            service.start(spec, tmp_path / "out")

    asyncio.run(go())


def test_an_unsupported_target_refuses_before_it_renders_the_wrong_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node offers targets the sweep cannot run yet. Falling back to FLUX.2 would produce a
    full report naming a model that never ran."""
    char = _character(tmp_path, monkeypatch, refs=2)
    from inline_core.studio.finetune import renderer_for

    service = finetune.CharacterTuning(
        _Events(), lambda target: renderer_for(target, None, None)
    )
    spec = finetune.SweepSpec(character=char, prompts=("a",), seeds=(1,), target="minimax")

    async def go() -> None:
        with pytest.raises(finetune.SweepRefused, match="do not run on"):
            service.start(spec, tmp_path / "out")

    asyncio.run(go())


def test_a_flagged_reference_is_re_tested_on_seeds_the_sweep_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    char = _character(tmp_path, monkeypatch, refs=3)
    seen: list[tuple[tuple[int, ...], int]] = []

    def render(req: finetune.CellRequest) -> Path:
        seen.append((req.select, req.seed))
        req.out.parent.mkdir(parents=True, exist_ok=True)
        req.out.write_bytes(b"x")
        return req.out

    # Reference 1 is the bad one: every render without it scores 20 points higher.
    def score(path: Path, chosen: str) -> dict[str, float]:
        select = seen[-1][0]
        return {"score": 30.0 if 1 in select else 50.0}

    events = _Events()
    service = finetune.CharacterTuning(events, lambda _t: render)
    monkeypatch.setattr(service, "_score", score)
    spec = finetune.SweepSpec(character=char, prompts=("a",), seeds=(1, 2), confirm_seeds=2)

    run_id = asyncio.run(_run(service, spec, tmp_path / "out"))
    result = service.result(run_id)

    confirmations = result["confirmations"]
    assert [c["index"] for c in confirmations] == [1]
    assert confirmations[0]["verdict"] == "removal confirmed"
    assert confirmations[0]["retest"] == -20.0
    # Two passes' pairs together, never just the re-test's.
    assert confirmations[0]["pooledPairs"] == 4
    # Fresh seeds only: a repeat at a seed already rendered is the same image.
    assert {seed for select, seed in seen if seed > 2} == {3, 4}
    # One shared full render per column, not one per flagged reference.
    assert result["done"] == 4 * 1 * 2 + 1 * 2 * 2 == result["total"]


def test_nothing_flagged_means_no_re_test_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    char = _character(tmp_path, monkeypatch, refs=2)

    def render(req: finetune.CellRequest) -> Path:
        req.out.parent.mkdir(parents=True, exist_ok=True)
        req.out.write_bytes(b"x")
        return req.out

    events = _Events()
    service = finetune.CharacterTuning(events, lambda _t: render)
    monkeypatch.setattr(service, "_score", lambda path, chosen: {"score": 60.0})
    spec = finetune.SweepSpec(character=char, prompts=("a",), seeds=(1,), confirm_seeds=2)

    run_id = asyncio.run(_run(service, spec, tmp_path / "out"))
    result = service.result(run_id)

    assert result["confirmations"] == []
    assert result["done"] == 3 == result["total"]


def test_a_finished_sweep_stores_the_report_path_it_can_be_opened_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read back after a restart, a run must still offer the report sitting beside it."""
    char = _character(tmp_path, monkeypatch, refs=2)
    out = tmp_path / "project" / "tuning_runs"

    def render(req: finetune.CellRequest) -> Path:
        req.out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (200, 120, 90)).save(req.out)
        return req.out

    events = _Events()
    service = finetune.CharacterTuning(events, lambda _t: render, lambda: out)
    monkeypatch.setattr(service, "_score", lambda path, chosen: {"score": 55.0})
    spec = finetune.SweepSpec(
        character=char, prompts=("a",), seeds=(1,), confirm_seeds=0, export_html=True
    )

    run_id = asyncio.run(_run(service, spec, out))

    stored = json.loads((out / run_id / "result.json").read_text())
    # Project-relative, because `/media` serves the project folder and cannot open an absolute path.
    assert stored["report"].startswith("tuning_runs/outputs/")
    assert (out.parent / stored["report"]).is_file()


def test_the_refusal_names_the_render_count_whether_or_not_it_re_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves of the sentence, both ways round: one branch used to lose the count."""
    char = _character(tmp_path, monkeypatch, refs=5)
    service = finetune.CharacterTuning(_Events(), lambda _t: (lambda req: req.out))

    for confirm_seeds, retest in ((0, False), (2, True)):
        spec = finetune.SweepSpec(
            character=char, prompts=("a",) * 6, seeds=(1, 2), confirm_seeds=confirm_seeds
        )
        with pytest.raises(finetune.SweepRefused) as refused:
            service.start(spec, tmp_path / "out")
        message = str(refused.value)
        assert message.startswith("That is 72 renders (6 reference sets x 6 prompts x 2 seeds)")
        assert message.count("(") == message.count(")")
        assert ("re-test" in message) is retest


def test_the_report_is_told_which_gallery_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to read a per-cell `detail` that the result never carried, so it always said refs."""
    char = _character(tmp_path, monkeypatch, refs=2)

    def render(req: finetune.CellRequest) -> Path:
        req.out.parent.mkdir(parents=True, exist_ok=True)
        req.out.write_bytes(b"x")
        return req.out

    service = finetune.CharacterTuning(_Events(), lambda _t: render)
    monkeypatch.setattr(
        service, "_score", lambda path, chosen: {"score": 60.0, "gallery": "frozen"}
    )
    spec = finetune.SweepSpec(character=char, prompts=("a",), seeds=(1,), confirm_seeds=0)

    run_id = asyncio.run(_run(service, spec, tmp_path / "out"))

    assert service.result(run_id)["gallery"] == "frozen"
