"""How the worker process group is launched. The transport and protocol do not care which.

DirectLauncher runs one process (rank 0) for a single GPU, CPU, or the stub test. TorchrunLauncher
starts N ranks via torchrun for a real multi-GPU split; its exact flags are finalized against the
2-GPU box when the xfuser handler lands (C2). The manager selects by world size.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from .config import ParallelConfig

_WORKER_MODULE = "inline_core.parallel.worker"


class Launcher(ABC):
    @abstractmethod
    def command(self, config: ParallelConfig) -> list[str]:
        """The argv that starts the worker process group."""


class DirectLauncher(Launcher):
    def command(self, config: ParallelConfig) -> list[str]:
        return [sys.executable, "-m", _WORKER_MODULE]


class TorchrunLauncher(Launcher):
    def command(self, config: ParallelConfig) -> list[str]:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={config.world_size}",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0",
            "-m",
            _WORKER_MODULE,
        ]


def default_launcher(config: ParallelConfig) -> Launcher:
    return TorchrunLauncher() if config.world_size > 1 else DirectLauncher()
