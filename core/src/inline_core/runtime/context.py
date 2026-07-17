"""The per-run execution context threaded through every component call."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

from ..device.policy import DevicePolicy
from .progress import ProgressEmitter


class CancelToken:
    """Cooperative cancellation. The executor checks it between nodes and steps."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class ExecutionContext:
    run_id: str
    policy: DevicePolicy
    emitter: ProgressEmitter
    cancel: CancelToken
