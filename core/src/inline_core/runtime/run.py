"""Run state: the durable snapshot GET /v1/runs/{id} serves, kept fresh by applying run events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..takes import Take
from .progress import (
    CancelledEvent,
    ErrorEvent,
    NodeDoneEvent,
    ProgressEmitter,
    ProgressEvent,
    RunDoneEvent,
    RunEvent,
)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class NodeState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CACHED = "cached"
    DONE = "done"
    ERROR = "error"


@dataclass
class NodeRuntimeState:
    state: NodeState = NodeState.QUEUED
    phase: str | None = None
    fraction: float = 0.0
    step: int | None = None
    step_count: int | None = None
    status: str = ""


@dataclass
class RunError:
    message: str
    node_id: str | None = None


@dataclass
class RunState:
    run_id: str
    target: str
    status: RunStatus = RunStatus.QUEUED
    fraction: float = 0.0
    nodes: dict[str, NodeRuntimeState] = field(default_factory=dict)
    takes: list[Take] = field(default_factory=list)
    error: RunError | None = None


_COMPLETE = (NodeState.DONE, NodeState.CACHED)


def _recompute_fraction(state: RunState) -> None:
    if not state.nodes:
        state.fraction = 0.0
        return
    total = sum(1.0 if n.state in _COMPLETE else n.fraction for n in state.nodes.values())
    state.fraction = total / len(state.nodes)


def apply_event(state: RunState, event: RunEvent) -> None:
    """Fold a run event into the run state. The single event -> snapshot mapping."""
    if isinstance(event, ProgressEvent):
        node = state.nodes.setdefault(event.node_id, NodeRuntimeState())
        node.state = NodeState.RUNNING
        node.phase = event.phase.value
        node.fraction = event.fraction
        node.step = event.step
        node.step_count = event.step_count
        node.status = event.status
        _recompute_fraction(state)
    elif isinstance(event, NodeDoneEvent):
        node = state.nodes.setdefault(event.node_id, NodeRuntimeState())
        node.state = NodeState.CACHED if event.cached else NodeState.DONE
        node.fraction = 1.0
        known = {t.id for t in state.takes}
        state.takes.extend(t for t in event.takes if t.id not in known)
        _recompute_fraction(state)
    elif isinstance(event, RunDoneEvent):
        state.status = RunStatus.DONE
        state.fraction = 1.0
    elif isinstance(event, CancelledEvent):
        state.status = RunStatus.CANCELLED
    elif isinstance(event, ErrorEvent):
        state.status = RunStatus.ERROR
        state.error = RunError(message=event.message, node_id=event.node_id)
        if event.node_id is not None:
            state.nodes.setdefault(event.node_id, NodeRuntimeState()).state = NodeState.ERROR


class StateTrackingEmitter(ProgressEmitter):
    """Applies each event to a RunState, then forwards it to the real emitter."""

    def __init__(self, delegate: ProgressEmitter, state: RunState) -> None:
        self._delegate = delegate
        self._state = state

    def emit(self, event: RunEvent) -> None:
        apply_event(self._state, event)
        self._delegate.emit(event)
