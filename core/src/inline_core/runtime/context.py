"""The per-run execution context threaded through every component call."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

from ..device.policy import DevicePolicy
from .progress import ProgressEmitter

if TYPE_CHECKING:
    from .store import TakeStore


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
    #: Where a node writes its output. Built-in runners hold their own store from construction;
    #: this is how a node built by the extension registrar (which calls `cls()`) reaches one.
    takes: TakeStore | None = None
