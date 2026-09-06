"""Reference sweeps: render a character many ways, score each render, report which refs earn a slot.

Never through ``RunManager``. A node runner occupies the single worker thread, so a nested submit
would queue behind the runner waiting for it and deadlock; this owns a worker of its own and calls
model runners directly, the way ``scripts/character_bench.py`` does.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..characters import apply as characters_apply
from ..characters import charfile as cf
from ..characters import encode, library, sweep

logger = logging.getLogger("inline_core.studio.finetune")

PROGRESS_EVENT = "events:tuneProgress"
LOG_EVENT = "events:tuneLog"
DONE_EVENT = "events:tuneDone"
ERROR_EVENT = "events:tuneError"

#: A sweep is combinations x prompts x seeds; the product runs away quickly, and every cell is a
#: full render. Past this the user is asked to confirm rather than discovering it hours later.
CONFIRM_ABOVE_CELLS = 60

#: Stop when the best adjusted score has not moved by this much over this many columns. The rest of
#: the sweep is then paying for renders that are not changing the answer.
PLATEAU_WINDOW = 4
PLATEAU_DELTA = 0.5

#: What one render costs on a paid target, so a sweep can be stopped by budget rather than by count.
#: Local targets cost GPU time only and carry no rate.
RATE_PER_CELL: dict[str, float] = {"flux2": 0.0, "minimax": 0.0, "fal-ref": 0.15}

#: What one cell renders with. The service owns nothing about a model beyond this.
Render = Callable[["CellRequest"], Path]


@dataclass(frozen=True)
class CellRequest:
    """One render: which references, which prompt, which seed."""

    character: str
    select: tuple[int, ...]
    prompt: str
    seed: int
    out: Path


@dataclass
class SweepSpec:
    character: str
    prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    target: str = "flux2"
    confirm: bool = False
    export_path: str = "outputs"
    #: Fresh seeds to re-test each reference the sweep flags for removal on. 0 skips the pass.
    confirm_seeds: int = 2
    export_html: bool = True
    #: Stops the sweep when the estimated spend reaches it. 0 means no ceiling, which is the right
    #: default only because the local targets are free.
    max_spend: float = 0.0


@dataclass
class SweepRun:
    id: str
    spec: SweepSpec
    refs: int
    combinations: list[str]
    cells: list[sweep.Cell] = field(default_factory=list)
    #: Kept out of `cells` so the sweep stays balanced: these columns hold two combinations, and
    #: folding them in would drop them from `balanced_cells` anyway.
    confirmations: list[sweep.Cell] = field(default_factory=list)
    confirm_total: int = 0
    status: str = "queued"
    error: str = ""
    report: str = ""
    warnings: list[str] = field(default_factory=list)
    started_at: int = 0
    ended_at: int = 0

    @property
    def total(self) -> int:
        sweep_cells = len(self.combinations) * len(self.spec.prompts) * len(self.spec.seeds)
        return sweep_cells + self.confirm_total

    @property
    def done(self) -> int:
        return len(self.cells) + len(self.confirmations)

    @property
    def spent(self) -> float:
        """Estimated, not billed: nothing reads a provider's invoice back."""
        return round(self.done * RATE_PER_CELL.get(self.spec.target, 0.0), 2)


class SweepRefused(ValueError):
    """The sweep cannot start, and says why rather than failing on the first render."""


def _current_loop() -> Any:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def encode_arch(target: str) -> str:
    """Which payload a target reads. The sweep must ask for the same one the render will."""
    return {
        "flux2": encode.FLUX2_KLEIN_ARCH,
        "minimax": encode.MINIMAX_H3_ARCH,
        "fal-ref": encode.FAL_REF_ARCH,
    }.get(target, encode.FLUX2_KLEIN_ARCH)


def preflight(spec: SweepSpec) -> tuple[int, list[str]]:
    """How many references a sweep runs over, and what it left out. Refuses rather than failing
    hours in: a character deleted since the graph was drawn, or references edited out of it."""
    path = library.resolve(spec.character)
    if path is None:
        raise SweepRefused(
            f"Character {spec.character!r} is no longer in models/characters/, so there is nothing "
            "to sweep. Pick another in the node's settings."
        )
    doc = cf.read(path)
    warnings: list[str] = []
    present = [r for r in doc.manifest.refs if doc.members.get(str(r.get("path"))) is not None]
    missing = len(doc.manifest.refs) - len(present)
    if missing:
        warnings.append(
            f"{missing} reference(s) are named by {path.name} but not inside it, so the sweep runs "
            "on what is left."
        )
    if len(present) < sweep.MIN_REFS:
        raise SweepRefused(
            f"{path.name} has {len(present)} usable reference(s). A sweep leaves one out at a "
            f"time, so it needs at least {sweep.MIN_REFS}."
        )
    if not spec.prompts:
        raise SweepRefused("A sweep needs at least one prompt to render.")
    if not spec.seeds:
        raise SweepRefused("A sweep needs at least one seed; a repeat at one seed re-renders.")
    return len(present), warnings


class CharacterTuning:
    """Runs reference sweeps off the event loop, one at a time."""

    def __init__(
        self,
        events: Any,
        render_for: Callable[[str], Render],
        root: Callable[[], Path] | None = None,
    ) -> None:
        self._events = events
        #: Keyed by target, so a sweep can never render a model other than the one it reports.
        self._render_for = render_for
        #: Where finished runs are written, so one can be read back after a restart.
        self._root = root
        self._runs: dict[str, SweepRun] = {}
        #: The sweep runs on a worker thread, and an asyncio queue may only be fed from the loop
        #: that owns it - the same reason `RunManager` hands its events over with
        #: `call_soon_threadsafe`. Without this every progress line was dropped in silence.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cancelled: set[str] = set()
        # One worker of its own: the graph's single worker is busy holding the node that started us.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tune")

    # --- reads ------------------------------------------------------------------------------

    def status(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None:
            raise ValueError(f"No sweep {run_id!r}.")
        return {
            "id": run.id,
            "status": run.status,
            "error": run.error,
            "done": run.done,
            "total": run.total,
            "warnings": list(run.warnings),
        }

    def result(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None:
            # Not in this process. A finished sweep keeps its findings beside its renders, which is
            # what lets a node still report them after the server restarts.
            stored = self._stored_result(run_id)
            if stored is not None:
                return stored
            raise ValueError(f"No sweep {run_id!r}.")
        found = sweep.contributions(run.cells, run.refs)
        ranked = sweep.combination_scores(run.cells)
        review = [c.index for c in found if c.verdict != "keep"]
        confirmed = self._confirmations(run)
        return {
            **self.status(run_id),
            "character": run.spec.character,
            "target": run.spec.target,
            "refs": run.refs,
            "gallery": self._gallery_of(run),
            "report": run.report,
            "spent": run.spent,
            # The one line the node face shows when it is done.
            "headline": (
                f"{ranked[0].key} at {ranked[0].adjusted:.0f}"
                f" · {len(review)} ref(s) to review" if ranked else "nothing measured"
            ),
            "caveat": (
                "face consistency, and wardrobe where the clothes are in frame; body shape is "
                "not scored"
            ),
            "combinations": [
                {
                    "key": c.key,
                    "adjusted": c.adjusted,
                    "mean": c.mean,
                    "spread": c.spread,
                    "tested": c.tested,
                }
                for c in ranked
            ],
            "contributions": [
                {
                    "index": c.index,
                    "delta": c.delta,
                    "spread": c.spread,
                    "pairs": c.pairs,
                    "verdict": c.verdict,
                    "perPrompt": c.per_prompt,
                    "framingShift": c.framing_shift,
                    "wardrobeDelta": c.wardrobe_delta,
                    "wardrobePairs": c.wardrobe_pairs,
                }
                for c in found
            ],
            "confirmations": [
                {
                    "index": c.index,
                    "initial": c.initial,
                    "retest": c.retest,
                    "spread": c.retest_spread,
                    "pairs": c.retest_pairs,
                    "pooled": c.pooled,
                    "pooledPairs": c.pooled_pairs,
                    "agrees": c.agrees,
                    "verdict": c.verdict,
                }
                for c in confirmed
            ],
            "cells": [
                {
                    "combination": c.combination,
                    "prompt": c.prompt,
                    "seed": c.seed,
                    "score": c.score,
                    "path": c.path,
                    "faceFraction": c.face_fraction,
                    "wardrobe": c.wardrobe,
                }
                for c in run.cells
            ],
        }

    # --- writes -----------------------------------------------------------------------------

    def _stored_result(self, run_id: str) -> dict[str, Any] | None:
        if self._root is None:
            return None
        path = self._root() / run_id / "result.json"
        if not path.is_file():
            return None
        try:
            return dict(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001 - a corrupt file reads as no result, never as a crash
            return None

    def start(self, spec: SweepSpec, out_dir: Path) -> dict[str, Any]:
        refs, warnings = preflight(spec)
        # Resolved here so an unsupported target refuses before a single render, not after twelve.
        self._render_for(spec.target)
        combinations = sweep.leave_one_out(refs)
        run = SweepRun(
            id=f"tune_{uuid.uuid4().hex[:12]}",
            spec=spec,
            refs=refs,
            combinations=combinations,
            warnings=warnings,
        )
        if run.total > CONFIRM_ABOVE_CELLS and not spec.confirm:
            counted = (
                f"That is {run.total} renders ({len(combinations)} reference sets x "
                f"{len(spec.prompts)} prompts x {len(spec.seeds)} seeds)"
            )
            retest = ""
            if spec.confirm_seeds:
                retest = ", plus a short re-test of any reference it flags"
            raise SweepRefused(
                f"{counted}{retest}. Tick Confirm to run it, or cut the prompts or seeds."
            )
        self._runs[run.id] = run
        # Captured here because `start` is the one method guaranteed to be on the loop thread.
        self._loop = asyncio.get_running_loop()
        for warning in warnings:
            self._log(run.id, f"warning: {warning}")
        asyncio.get_running_loop().run_in_executor(self._pool, self._sweep, run, out_dir)
        return self.status(run.id)

    def cancel(self, run_id: str) -> None:
        """Stops between cells, so whatever finished stays a balanced result."""
        self._cancelled.add(run_id)

    # --- the loop ---------------------------------------------------------------------------

    def _sweep(self, run: SweepRun, out_dir: Path) -> None:
        run.status = "running"
        run.started_at = int(time.time() * 1000)
        self._log(run.id, f"sweeping {run.refs} references over {run.total} renders")
        try:
            # (prompt, seed) outer, combination inner: every finished column is a complete block,
            # so cancelling half way still leaves a result the paired maths can read.
            for prompt in run.spec.prompts:
                for seed in run.spec.seeds:
                    for key in run.combinations:
                        if run.id in self._cancelled:
                            self._finish(run, "cancelled", out_dir)
                            return
                        self._cell(run, key, prompt, seed, out_dir)
                    stop = self._should_stop(run)
                    if stop:
                        self._log(run.id, f"stopping early: {stop}")
                        self._confirm_pass(run, out_dir)
                        self._finish(run, "done", out_dir)
                        return
            self._confirm_pass(run, out_dir)
            self._finish(run, "done", out_dir)
        except Exception as error:  # noqa: BLE001 - a sweep reports its failure, never crashes out
            logger.exception("Sweep %s failed", run.id)
            run.error = str(error)
            self._finish(run, "error", out_dir)

    def _cell(
        self,
        run: SweepRun,
        key: str,
        prompt: str,
        seed: int,
        out_dir: Path,
        into: list[sweep.Cell] | None = None,
    ) -> None:
        select = tuple(sweep.selection_for(key, run.refs))
        stage = "confirm" if into is not None else key
        prefix = f"{key}-" if into is not None else ""
        target = out_dir / run.id / stage / f"{prefix}{abs(hash(prompt)) % 10**8}-{seed}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        cell = sweep.Cell(combination=key, prompt=prompt, seed=seed)
        try:
            path = self._render_for(run.spec.target)(
                CellRequest(
                    character=run.spec.character,
                    select=select,
                    prompt=prompt,
                    seed=seed,
                    out=target,
                )
            )
            cell.path = str(path)
            scored = self._score(path, run.spec.character)
            if scored is not None:
                cell.score = float(scored["score"])
                cell.detail = scored
                cell.face_fraction = self._face_fraction(path)
                # Only when the clothes were in frame: a close-up scores low on wardrobe because
                # there are no clothes in it, and differencing that against a full body is noise.
                if scored.get("wardrobeCounted"):
                    cell.wardrobe = float(scored["wardrobeScore"])
        except Exception as error:  # noqa: BLE001 - one failed render drops its column, not the run
            self._log(run.id, f"{key} · seed {seed} · failed: {error}")
        (run.cells if into is None else into).append(cell)
        shown = "unmeasurable" if cell.score is None else f"{cell.score:.1f}"
        self._log(run.id, f"{key} · seed {seed} · {shown}")
        self._emit(
            PROGRESS_EVENT,
            {
                "runId": run.id,
                "done": run.done,
                "total": run.total,
                "fraction": run.done / run.total,
            },
        )

    def _confirm_pass(self, run: SweepRun, out_dir: Path) -> None:
        """Re-run the full set against each flagged reference on seeds the sweep never used.

        A repeat at a seed the sweep already rendered is the same image, so only new seeds add a
        pair. Only the flagged references are re-tested: confirming a drop is what changes what the
        user does, and re-testing everything would cost as much as the sweep again.
        """
        if not run.spec.confirm_seeds or run.id in self._cancelled:
            return
        marked = sweep.flagged(sweep.contributions(run.cells, run.refs))
        if not marked:
            self._log(run.id, "no reference was flagged for removal, so nothing to confirm")
            return
        base = max(run.spec.seeds) + 1
        seeds = [base + i for i in range(run.spec.confirm_seeds)]
        # One shared full render per column rather than one per flagged reference: it is the same
        # image at the same seed, and every drop is differenced against it.
        run.confirm_total = len(run.spec.prompts) * len(seeds) * (1 + len(marked))
        names = ", ".join(f"ref {c.index} ({c.delta:+.1f})" for c in marked)
        self._log(
            run.id,
            f"confirming {names} on {len(seeds)} fresh seed(s): {run.confirm_total} more renders",
        )
        keys = [sweep.FULL, *(sweep.combination_key(c.index) for c in marked)]
        for prompt in run.spec.prompts:
            for seed in seeds:
                for key in keys:
                    if run.id in self._cancelled:
                        return
                    self._cell(run, key, prompt, seed, out_dir, into=run.confirmations)

    def _gallery_of(self, run: SweepRun) -> str:
        """Which gallery answered. Read off the cells, which keep the whole scoring result."""
        for cell in run.cells:
            named = cell.detail.get("gallery")
            if named:
                return str(named)
        return "refs"

    def _confirmations(self, run: SweepRun) -> list[sweep.Confirmation]:
        found = {c.index: c.delta for c in sweep.contributions(run.cells, run.refs)}
        # Numerically, not by key: sorting the strings puts drop_10 ahead of drop_2.
        indices = sorted(
            int(key.removeprefix("drop_"))
            for key in {c.combination for c in run.confirmations} - {sweep.FULL}
        )
        return [
            sweep.confirm(index, found.get(index, 0.0), run.cells, run.confirmations)
            for index in indices
        ]

    def _should_stop(self, run: SweepRun) -> str:
        """Why the sweep should stop before its last column, or an empty string to keep going.

        Checked between columns rather than between cells, so a stop still leaves whole blocks.
        """
        if run.spec.max_spend and run.spent >= run.spec.max_spend:
            return f"estimated spend ${run.spent:.2f} reached the ${run.spec.max_spend:.2f} cap"
        if sweep.plateaued(run.cells, PLATEAU_WINDOW, PLATEAU_DELTA):
            return (
                f"the best set moved less than {PLATEAU_DELTA} over {PLATEAU_WINDOW} columns, so "
                "more renders are not changing the answer"
            )
        return ""

    def _score(self, path: Path, chosen: str) -> dict[str, Any] | None:
        from .characters import Characters

        return Characters(store=None, events=None).score_take(path, chosen)

    def _face_fraction(self, path: Path) -> float | None:
        """How much of the frame the face fills. Reported because the score follows it."""
        from PIL import Image

        from ..characters import scoring

        try:
            with Image.open(path) as handle:
                return scoring.face_fraction(handle.convert("RGB"))
        except Exception:  # noqa: BLE001 - a missing framing number is not worth failing a cell
            return None

    def _finish(self, run: SweepRun, status: str, out_dir: Path) -> None:
        run.ended_at = int(time.time() * 1000)
        self._cancelled.discard(run.id)
        if status == "error":
            run.status = status
            self._emit(ERROR_EVENT, {"runId": run.id, "error": run.error})
            self._log(run.id, f"failed: {run.error}")
            return
        # The status flips last, below: a caller that polls it and then reads the run must not be
        # told the sweep finished before the files it is about to look for exist.
        # Written before the report, so a failed report still leaves the findings readable.
        self._persist(run, out_dir, status)
        if run.spec.export_html and run.cells:
            try:
                # Stored project-relative: the client opens it through `/media`, which serves the
                # project folder, and an absolute path is not reachable from a browser.
                run.report = self._project_relative(self._write_report(run, out_dir))
            except Exception as error:  # noqa: BLE001 - a report must not lose a finished sweep
                self._log(run.id, f"warning: the report could not be written: {error}")
            else:
                # Rewritten now the path is known, or a sweep read back after a restart offers no
                # link to the report sitting beside it.
                self._persist(run, out_dir, status)
                self._log(run.id, f"report: {run.report}")
        run.status = status
        self._log(run.id, f"{status}: {run.done}/{run.total} renders")
        self._emit(
            DONE_EVENT, {"runId": run.id, "status": status, "result": self.result(run.id)}
        )

    def _persist(self, run: SweepRun, out_dir: Path, status: str) -> None:
        try:
            target = out_dir / run.id / "result.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            # The status is passed in because the run is not marked finished until the files are.
            target.write_text(json.dumps({**self.result(run.id), "status": status}, indent=2))
        except Exception:  # noqa: BLE001 - losing durability must not lose the run in flight
            logger.warning("Could not persist the result for %s", run.id)

    def _project_relative(self, path: Path) -> str:
        root = self._root().parent if self._root is not None else None
        try:
            return str(path.relative_to(root)) if root else str(path)
        except ValueError:
            # Exported somewhere outside the project, so it is only openable on the machine.
            return str(path)

    def _write_report(self, run: SweepRun, out_dir: Path) -> Path:
        from .tuning_report import build_report

        where = Path(run.spec.export_path).expanduser()
        if not where.is_absolute():
            where = out_dir / where
        result = self.result(run.id)
        result["referenceThumbs"] = self._reference_thumbs(run)
        return build_report(result, where)

    def _reference_thumbs(self, run: SweepRun) -> dict[str, str]:
        """The references themselves, embedded, so a verdict names a picture and not an index."""
        from .tuning_report import embed_image

        applied = characters_apply.char_apply(
            run.spec.character, encode_arch(run.spec.target), prefer="reference"
        )
        if applied is None:
            return {}
        return {str(i): embed_image(str(ref.path)) for i, ref in enumerate(applied.refs)}

    def _log(self, run_id: str, line: str) -> None:
        self._emit(LOG_EVENT, {"runId": run_id, "line": line})

    def _emit(self, channel: str, payload: dict[str, Any]) -> None:
        """Hand an event to the loop thread, wherever it was raised."""
        loop = self._loop
        if loop is None or loop == _current_loop():
            self._events.broadcast(channel, payload)
            return
        loop.call_soon_threadsafe(self._events.broadcast, channel, payload)


def renderer_for(target: str, store: Any, policy: Any) -> Render:
    """The renderer a target actually uses.

    An unimplemented target raises rather than falling back: a sweep that silently rendered FLUX.2
    while the node said MiniMax would produce a full report about the wrong model.
    """
    if target == "flux2":
        return flux2_render(store, policy)
    raise SweepRefused(
        f"Sweeps do not run on {target!r} yet. Pick FLUX.2, which is the target this measures."
    )


def flux2_render(store: Any, policy: Any) -> Render:
    """One FLUX.2 image per cell, through the shipped node.

    The character is *wired*, never a param - FLUX.2 declares no such param, and passing one is how
    `character_bench.py` rendered unconditioned images for months.
    """

    def render(request: CellRequest) -> Path:
        from ..graph.schema import Node
        from ..models.flux2.runner import FLUX2, Flux2Runner
        from ..runtime.context import CancelToken, ExecutionContext
        from ..runtime.progress import NullEmitter

        path = library.resolve(request.character)
        if path is None:
            raise SweepRefused(f"Character {request.character!r} vanished mid-sweep.")
        identity = _Selected(doc=cf.read(path), file=path.name, select=request.select)
        node = Node(id="tune", type=FLUX2.type, params={"seed": request.seed})
        ctx = ExecutionContext(
            run_id=f"tune-{request.seed}",
            policy=policy,
            emitter=NullEmitter(),
            cancel=CancelToken(),
        )
        result = Flux2Runner(store, policy).run(
            node, {"prompt": [request.prompt], "character": [identity]}, ctx
        )
        request.out.write_bytes(Path(result.takes[0].uri).read_bytes())
        return request.out

    return render


@dataclass
class _Selected:
    """An identity carrying which of its references to send, read by the runners' `select` hook."""

    doc: Any
    file: str
    select: tuple[int, ...]


__all__ = [
    "CellRequest",
    "CharacterTuning",
    "SweepRefused",
    "SweepSpec",
    "flux2_render",
    "renderer_for",
    "preflight",
    "characters_apply",
]
