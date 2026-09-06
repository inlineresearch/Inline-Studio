"""``character/finetune``: the node that drives a reference sweep.

It does not render. The sweep lives out of the graph, because the alternatives are both impossible:
a node that wires to a generation node and reads its take back is a cycle, and a runner that submits
to the RunManager and waits deadlocks on the single worker it is itself occupying. So this starts a
durable sweep and blocks on it, exactly as ``train/lora`` drives the training service.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from ...errors import ComponentError
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...runtime.context import ExecutionContext
from ...studio.finetune import SweepRefused, SweepSpec

logger = logging.getLogger("inline_core.models.character.finetune")

POLL_SECONDS = 1.0
_TERMINAL = ("done", "error", "cancelled")

#: Ten prompts that move everything except identity - shot size, light, setting, wardrobe, motion.
#: A character that only holds on head-and-shoulders is not holding, and an easier set hides that.
#: Reordered so the first four span shot size, which is the axis the score follows.
PROMPT_LIBRARY = (
    "<char>, a tight close-up portrait, soft window light",
    "<char>, a full body shot standing in an empty warehouse, wide angle",
    "<char> walking down a busy street in the rain, overcast daylight",
    "<char> sitting in a dark bar lit by a single red neon sign",
    "<char>, a side profile against a plain grey wall",
    "<char> wearing a heavy winter coat and scarf in the snow",
    "<char> running across a rooftop at sunset, motion blur",
    "<char> sitting at a dinner table talking with two other people",
    "<char> photographed from a low angle looking up, dramatic sky",
    "<char>, a cinematic film still, 35mm grain, muted teal and orange grade",
)

#: Four, not the whole library: ten prompts at two seeds is 140 renders on a six-reference
#: character, over the confirm gate, and a node that refuses on first use teaches nothing. Two of
#: the four are full-body so the wardrobe term has cells where the clothes are in frame at all.
DEFAULT_PROMPTS = PROMPT_LIBRARY[:4]

#: More than this and the sweep is long enough that the prompt set is the wrong lever.
MAX_PROMPTS = 20

FINETUNE = NodeDescriptor(
    type="character/finetune",
    title="Finetune Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(Port("character", "Character", PortKind.CHARACTER, required=True),),
    # The character passes through so the node chains; `metrics` carries the run id, which is what
    # a Logger wired to it reads its stream by.
    outputs=(
        Port("character", "Character", PortKind.CHARACTER),
        Port("metrics", "Log", PortKind.METRICS),
    ),
    params=(
        ParamField(
            "target", "Model", Widget.SELECT, "flux2",
            options=(
                Option(value="flux2", label="FLUX.2"),
                Option(value="minimax", label="MiniMax H3"),
            ),
        ),
        # `<char>` rather than a name, so one prompt set moves between characters unchanged.
        ParamField(
            "prompts", "Test prompts (one per line, <char> is the subject)",
            Widget.TEXTAREA, "\n".join(DEFAULT_PROMPTS), on_face=True,
        ),
        ParamField("seeds", "Seeds (comma separated)", Widget.TEXT, "11,22", on_face=True),
        ParamField(
            "export_path", "Export to", Widget.TEXT, "outputs", on_face=True, kind="file"
        ),
        ParamField(
            "confirm_seeds", "Re-test flagged references on N fresh seeds", Widget.NUMBER, 2,
            min=0, max=8, step=1,
        ),
        ParamField("export_html", "Write an HTML report", Widget.BOOLEAN, True),
        ParamField("confirm", "Confirm a long sweep", Widget.BOOLEAN, False),
    ),
)


class TuningBridge:
    """Runs a ``CharacterTuning`` call on the loop thread and blocks the worker until it returns."""

    def __init__(self, tuning: Any, on_bound: Callable[[str, str], None] | None = None) -> None:
        self.tuning = tuning
        self.on_bound = on_bound
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = self._loop
        # Unbound means no server is running it (tests drive the runner directly), so call in place.
        if loop is None:
            return fn(*args)

        async def invoke() -> Any:
            return fn(*args)

        return asyncio.run_coroutine_threadsafe(invoke(), loop).result()


def parse_prompts(raw: Any, subject: str = "") -> tuple[str, ...]:
    """One per line, `<char>` replaced by the character's own name so a set is portable."""
    lines = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
    if len(lines) > MAX_PROMPTS:
        raise ComponentError(
            f"{len(lines)} prompts is past the {MAX_PROMPTS} this node takes. Every prompt is a "
            "render for every reference set, so the sweep grows with each one."
        )
    return tuple(line.replace("<char>", subject or "the character") for line in lines)


def parse_seeds(raw: Any) -> tuple[int, ...]:
    """Distinct seeds. A repeat at one seed re-renders the same image and measures nothing."""
    out: list[int] = []
    for piece in str(raw or "").replace(" ", "").split(","):
        if not piece:
            continue
        try:
            seed = int(piece)
        except ValueError as error:
            raise ComponentError(f"{piece!r} is not a seed.") from error
        if seed not in out:
            out.append(seed)
    return tuple(out)


class FinetuneRunner(NodeRunner):
    """Starts a sweep and forwards its progress, then hands the character on unchanged."""

    def __init__(self, bridge: TuningBridge, out_dir: Callable[[], Any]) -> None:
        self._bridge = bridge
        self._out_dir = out_dir

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        wired = (inputs.get("character") or [None])[0]
        if wired is None:
            raise ComponentError("Finetune Character needs a character.")
        chosen = str(getattr(wired, "file", "") or "")
        if not chosen:
            raise ComponentError(
                "That character has not been saved yet. Wire it through Write .char first."
            )
        subject = str(getattr(wired, "name", "") or "") or chosen.removesuffix(".char")
        # Under the descriptor's defaults: a node the user never edited stores no params at all,
        # while its face shows the defaults it would use. Reading `node.params` alone made the node
        # look configured and then refuse for having no prompts.
        params = {**FINETUNE.defaults(), **node.params}
        spec = SweepSpec(
            character=chosen,
            prompts=parse_prompts(params.get("prompts"), subject),
            seeds=parse_seeds(params.get("seeds")),
            target=str(params.get("target") or "flux2"),
            confirm=bool(params.get("confirm")),
            confirm_seeds=max(0, int(params.get("confirm_seeds") or 0)),
            export_path=str(params.get("export_path") or "outputs"),
            export_html=bool(params.get("export_html", True)),
        )
        service = self._bridge.tuning
        try:
            started = self._bridge.call(service.start, spec, self._out_dir())
        except SweepRefused as refused:
            # A refusal is an answer, not a crash: it names what to change, and the node should
            # show that rather than a traceback.
            raise ComponentError(str(refused)) from refused
        run_id = str(started["id"])
        logger.info("Sweep %s started for %s", run_id, chosen)
        # The node binds to the run it started: that is what a wired Logger reads its lines by.
        if self._bridge.on_bound is not None:
            self._bridge.call(self._bridge.on_bound, node.id, run_id)
        state = self._await_run(run_id)
        if state["status"] == "error":
            raise ComponentError(state.get("error") or "The sweep failed.")
        return NodeResult(outputs={"character": wired, "metrics": run_id})

    def _await_run(self, run_id: str) -> dict[str, Any]:
        service = self._bridge.tuning
        while True:
            state = self._bridge.call(service.status, run_id)
            if str(state.get("status") or "") in _TERMINAL:
                return state
            time.sleep(POLL_SECONDS)


def register_finetune_node(registry: Any, tuning: Any, out_dir: Any, on_bound: Any = None) -> Any:
    bridge = TuningBridge(tuning, on_bound)
    registry.register(FINETUNE, FinetuneRunner(bridge, out_dir))
    return bridge
