"""Run events and the emitter seam. Mirrors the websocket events in docs/contract.md section 6.

Coalescing (the contract's bounded event rate) is a wrapper added at the serving layer; the executor
and components just emit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..takes import Take


class Phase(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    LOADING = "loading"
    ENCODE = "encode"
    SAMPLE = "sample"
    DECODE = "decode"
    SAVE = "save"


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    node_id: str
    phase: Phase
    fraction: float
    step: int | None = None
    step_count: int | None = None
    eta_ms: int | None = None
    status: str = ""


@dataclass(frozen=True)
class NodeDoneEvent:
    run_id: str
    node_id: str
    cached: bool
    takes: list[Take] = field(default_factory=list)


@dataclass(frozen=True)
class RunDoneEvent:
    run_id: str


@dataclass(frozen=True)
class CancelledEvent:
    run_id: str


@dataclass(frozen=True)
class ErrorEvent:
    run_id: str
    message: str
    node_id: str | None = None


RunEvent = ProgressEvent | NodeDoneEvent | RunDoneEvent | CancelledEvent | ErrorEvent


class ProgressEmitter(ABC):
    @abstractmethod
    def emit(self, event: RunEvent) -> None: ...


class NullEmitter(ProgressEmitter):
    def emit(self, event: RunEvent) -> None:
        return None


class CollectingEmitter(ProgressEmitter):
    """Keeps every event in order. For tests and the snapshot builder."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)
