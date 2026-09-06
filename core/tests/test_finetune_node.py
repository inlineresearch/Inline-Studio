"""The node half: what it parses, what it refuses, and that it blocks on the sweep it started."""

from __future__ import annotations

from typing import Any

import pytest

from inline_core.errors import ComponentError
from inline_core.graph.schema import Node
from inline_core.models.character import finetune as fn


class _Tuning:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.started: Any = None

    def start(self, spec: Any, out: Any) -> dict[str, Any]:
        self.started = spec
        return {"id": "tune_1", "status": "queued"}

    def status(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": self.statuses.pop(0), "error": "it broke"}


def _node(params: dict[str, Any]) -> Node:
    return Node(id="n1", type=fn.FINETUNE.type, params=params)


class _Wired:
    file = "Emmy.char"


def test_seeds_are_distinct_because_a_repeat_at_one_seed_re_renders() -> None:
    assert fn.parse_seeds("11, 22, 11") == (11, 22)
    assert fn.parse_seeds("") == ()
    with pytest.raises(ComponentError, match="not a seed"):
        fn.parse_seeds("11,abc")


def test_prompts_are_one_per_line_with_blanks_dropped() -> None:
    assert fn.parse_prompts("a\n\n  b  \n") == ("a", "b")


def test_the_node_refuses_an_unsaved_character() -> None:
    """Applying resolves payloads through a content-keyed cache, so it needs a file, not a doc."""
    bridge = fn.TuningBridge(_Tuning(["done"]))
    runner = fn.FinetuneRunner(bridge, lambda: "/tmp")

    class _Unsaved:
        file = ""

    with pytest.raises(ComponentError, match="Write .char first"):
        runner.run(_node({}), {"character": [_Unsaved()]}, None)  # type: ignore[arg-type]
    with pytest.raises(ComponentError, match="needs a character"):
        runner.run(_node({}), {}, None)  # type: ignore[arg-type]


def test_the_node_blocks_until_the_sweep_is_terminal_then_passes_the_character_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocking is what keeps a graph Run sequential, the same as Train LoRA."""
    monkeypatch.setattr(fn, "POLL_SECONDS", 0)
    tuning = _Tuning(["running", "running", "done"])
    bound: list[tuple[str, str]] = []
    bridge = fn.TuningBridge(tuning, lambda item, run: bound.append((item, run)))
    wired = _Wired()

    result = fn.FinetuneRunner(bridge, lambda: "/tmp").run(
        _node({"prompts": "walking\nsitting", "seeds": "1,2"}), {"character": [wired]}, None  # type: ignore[arg-type]
    )

    assert tuning.statuses == [], "it polled to a terminal state rather than returning early"
    # The Logger finds its stream by this binding; the graph's progress does not carry a run id.
    assert bound == [("n1", "tune_1")]
    assert result.outputs["metrics"] == "tune_1"
    assert result.outputs["character"] is wired, "the character chains on unchanged"
    assert tuning.started.prompts == ("walking", "sitting")
    assert tuning.started.seeds == (1, 2)


def test_a_failed_sweep_fails_the_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fn, "POLL_SECONDS", 0)
    bridge = fn.TuningBridge(_Tuning(["error"]))
    with pytest.raises(ComponentError, match="it broke"):
        fn.FinetuneRunner(bridge, lambda: "/tmp").run(
            _node({}), {"character": [_Wired()]}, None  # type: ignore[arg-type]
        )


def test_a_node_the_user_never_edited_runs_on_its_own_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freshly added node stores no params at all, while its face renders the defaults it would
    use. Reading `node.params` alone made the node look configured with ten prompts and then refuse
    the run for having none."""
    monkeypatch.setattr(fn, "POLL_SECONDS", 0)
    tuning = _Tuning(["done"])
    runner = fn.FinetuneRunner(fn.TuningBridge(tuning), lambda: "/tmp")

    runner.run(_node({}), {"character": [_Wired()]}, None)  # type: ignore[arg-type]

    assert len(tuning.started.prompts) == len(fn.DEFAULT_PROMPTS)
    assert tuning.started.seeds == (11, 22)
    assert tuning.started.target == "flux2"
    assert tuning.started.export_html is True
    # `<char>` resolves against the wired character, so the default set is portable.
    assert "<char>" not in tuning.started.prompts[0]
    assert "Emmy" in tuning.started.prompts[0]


def test_an_edited_param_still_wins_over_its_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fn, "POLL_SECONDS", 0)
    tuning = _Tuning(["done"])
    runner = fn.FinetuneRunner(fn.TuningBridge(tuning), lambda: "/tmp")

    runner.run(
        _node({"prompts": "just one", "seeds": "7"}), {"character": [_Wired()]}, None  # type: ignore[arg-type]
    )
    assert tuning.started.prompts == ("just one",)
    assert tuning.started.seeds == (7,)


def test_a_refusal_reaches_the_node_as_a_message_not_a_traceback() -> None:
    """A sweep that is too large, or a character that is gone, is an answer the user can act on.
    `SweepRefused` is a ValueError, so without this the executor logged it as an unhandled error
    and the canvas showed a stack trace instead of what to change."""
    from inline_core.studio.finetune import SweepRefused

    class _Refuses:
        def start(self, spec: Any, out: Any) -> dict[str, Any]:
            raise SweepRefused("That is 120 renders. Tick Confirm to run it.")

    runner = fn.FinetuneRunner(fn.TuningBridge(_Refuses()), lambda: "/tmp")
    with pytest.raises(ComponentError, match="Tick Confirm"):
        runner.run(_node({}), {"character": [_Wired()]}, None)  # type: ignore[arg-type]
