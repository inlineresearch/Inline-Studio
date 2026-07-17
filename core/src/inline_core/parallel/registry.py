"""One running worker group per parallel config, created lazily and reused across jobs.

The server owns a single registry and shuts it down with the app; the parallel sampler asks it for
the group that matches a job's placement. Keying on the frozen ParallelConfig means the same model
and split reuse one process group.
"""

from __future__ import annotations

from .config import ParallelConfig
from .group import WorkerGroup
from .launch import Launcher


class GroupRegistry:
    def __init__(self, launcher: Launcher | None = None) -> None:
        self._launcher = launcher
        self._groups: dict[ParallelConfig, WorkerGroup] = {}

    def get_or_create(self, config: ParallelConfig) -> WorkerGroup:
        group = self._groups.get(config)
        if group is None:
            group = WorkerGroup(config, self._launcher)
            group.start()
            self._groups[config] = group
        return group

    def shutdown_all(self) -> None:
        while self._groups:
            _, group = self._groups.popitem()
            group.shutdown()
