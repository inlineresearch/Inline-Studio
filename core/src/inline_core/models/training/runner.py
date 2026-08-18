"""LoRA training as graph nodes, so one Run walks dataset -> caption -> train in topological order.

The nodes do not reimplement training: they drive the durable ``Training`` service, which owns the
FIFO queue, the subprocess, SIGTERM cancel and the resumable checkpoint. That service and the
project DB both live on the server's event loop thread, while node runners execute on a worker, so
every call hops across through ``TrainingBridge``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...errors import ComponentError
from ...graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ..pipeline_runtime import progress_event

logger = logging.getLogger("inline_core.training")

#: How often the worker thread asks the loop how the run is doing. A training step is seconds long.
POLL_SECONDS = 1.0

#: Statuses a run can end on. `interrupted` is terminal for the graph but resumable for the user.
_DONE = "done"
_TERMINAL = (_DONE, "failed", "cancelled", "interrupted")


@dataclass
class Dataset:
    """A training dataset in flight. Carries the id because every service call is keyed by it."""

    id: str
    name: str


class TrainingBridge:
    """Runs a ``Training`` call on the loop thread and blocks the worker until it returns."""

    def __init__(self, training: Any, on_bound: Callable[[str, str], None] | None = None) -> None:
        self.training = training
        #: Told (node id, run id) when a training node starts one, so the canvas can bind to it.
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


LOAD_DATASET = NodeDescriptor(
    type="train/dataset",
    title="Load Dataset",
    category="Training",
    icon="layers",
    output_kind=None,
    hidden=True,
    inputs=(),
    outputs=(Port("dataset", "Dataset", PortKind.DATASET),),
    params=(ParamField("dataset_id", "Dataset", Widget.TEXT, ""),),
)

CAPTION = NodeDescriptor(
    type="train/caption",
    title="Caption",
    category="Training",
    icon="text",
    output_kind=None,
    hidden=True,
    inputs=(Port("dataset", "Dataset", PortKind.DATASET, required=True),),
    outputs=(Port("dataset", "Dataset", PortKind.DATASET),),
    params=(
        ParamField("overwrite", "Recaption everything", Widget.BOOLEAN, False),
        ParamField("captioner", "Captioner", Widget.TEXT, ""),
    ),
)

TRAIN_LORA = NodeDescriptor(
    type="train/lora",
    title="Train LoRA",
    category="Training",
    icon="wand",
    output_kind=None,
    hidden=True,
    inputs=(Port("dataset", "Dataset", PortKind.DATASET, required=True),),
    outputs=(Port("lora", "LoRA", PortKind.LORA),),
    # Hyperparams arrive as one blob: they are edited in the node's own Adjust sidebar, which knows
    # which fields a given architecture offers, not by the generic param renderer.
    params=(),
)


def register_training_nodes(
    registry: Any, training: Any, on_bound: Callable[[str, str], None] | None = None
) -> TrainingBridge:
    """Register the three executable training nodes. Returns the bridge for the server to bind."""
    bridge = TrainingBridge(training, on_bound)
    registry.register(LOAD_DATASET, LoadDatasetRunner(bridge))
    registry.register(CAPTION, CaptionRunner(bridge))
    registry.register(TRAIN_LORA, TrainLoraRunner(bridge))
    return bridge


def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


class LoadDatasetRunner(NodeRunner):
    """Names a dataset for the nodes downstream, and fails now rather than after a caption pass."""

    def __init__(self, bridge: TrainingBridge) -> None:
        self._bridge = bridge

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        dataset_id = str(node.params.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ComponentError("Load Dataset needs a dataset picked on the node.")
        datasets = self._bridge.call(self._bridge.training.list_datasets)
        found = next((d for d in datasets if d["id"] == dataset_id), None)
        if found is None:
            raise ComponentError("That dataset no longer exists.")
        return NodeResult(outputs={"dataset": Dataset(id=dataset_id, name=str(found["name"]))})


class CaptionRunner(NodeRunner):
    """Captions the dataset in place and passes it on, so Train LoRA reads the new text."""

    def __init__(self, bridge: TrainingBridge) -> None:
        self._bridge = bridge

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        dataset = _first(inputs.get("dataset"))
        if not isinstance(dataset, Dataset):
            raise ComponentError("Caption needs a dataset.")
        overwrite = bool(node.params.get("overwrite", False))
        captioner = str(node.params.get("captioner") or "") or None
        ctx.emitter.emit(
            progress_event(ctx, node, Phase.PREPARING, 0.0, status=f"Captioning {dataset.name}…")
        )
        self._bridge.call(self._bridge.training.auto_caption, dataset.id, overwrite, captioner)
        logger.info("Captioned dataset %s (overwrite=%s)", dataset.name, overwrite)
        return NodeResult(outputs={"dataset": dataset})


class TrainLoraRunner(NodeRunner):
    """Starts a durable training run and forwards its steps as this node's progress.

    The node blocks until the run reaches a terminal state, which is what makes a graph-wide Run
    sequential. Cancelling the graph SIGTERMs the trainer, which flushes a resumable checkpoint.
    """

    def __init__(self, bridge: TrainingBridge) -> None:
        self._bridge = bridge

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        dataset = _first(inputs.get("dataset"))
        if not isinstance(dataset, Dataset):
            raise ComponentError("Train LoRA needs a dataset.")
        hyperparams = dict(node.params.get("hyperparams") or {})
        service = self._bridge.training

        run = self._bridge.call(service.start, dataset.id, hyperparams)
        run_id = str(run["id"])
        logger.info("Training run %s started for dataset %s", run_id, dataset.name)
        # The node binds to the run it started: that is what Resume, the log tail and the loss
        # curve are keyed by, and the graph's own progress stream does not carry it.
        if self._bridge.on_bound is not None:
            self._bridge.call(self._bridge.on_bound, node.id, run_id)
        state = self._await_run(node, ctx, run_id)

        status = str(state.get("status") or "")
        if status != _DONE:
            raise ComponentError(
                str(state.get("error") or "") or f"Training {status or 'did not finish'}."
            )
        path = str(state.get("outputLoraPath") or "")
        if not path:
            raise ComponentError("Training finished without writing an adapter.")
        return NodeResult(outputs={"lora": path})

    def _await_run(self, node: Node, ctx: ExecutionContext, run_id: str) -> dict[str, Any]:
        """Poll until terminal, mirroring steps into the graph's progress stream."""
        service = self._bridge.training
        # Cancel is a SIGTERM the trainer answers by flushing a checkpoint, so the poll continues
        # afterwards rather than reporting a state the process has not reached yet. Local, because
        # the context is shared with every other node in the run.
        sent = False
        while True:
            if not sent and ctx.cancel.cancelled:
                self._bridge.call(service.cancel, run_id)
                sent = True
            state = self._bridge.call(service.status, run_id)
            if str(state.get("status") or "") in _TERMINAL:
                return state
            ctx.emitter.emit(_step_progress(ctx, node, state))
            time.sleep(POLL_SECONDS)


def _step_progress(ctx: ExecutionContext, node: Node, state: dict[str, Any]) -> Any:
    step = state.get("step")
    total = state.get("totalSteps")
    fraction = float(state.get("progressFraction") or 0.0)
    status = str(state.get("progressStatus") or state.get("status") or "")
    return progress_event(
        ctx, node, Phase.SAMPLE, fraction,
        step=int(step) if isinstance(step, int) else None,
        step_count=int(total) if isinstance(total, int) else None,
        status=status,
    )
