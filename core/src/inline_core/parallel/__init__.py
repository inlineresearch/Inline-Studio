"""Isolated xfuser worker group behind the batched-sampler seam.

A parallel denoise runs in a separate process group the engine spawns and talks to over a local
socket. The HTTP server, DB, graph, and orchestration stay single-process; only the denoise loop is
distributed. See PLAN.md (multi-GPU) and `sampling/batch.py` (the seam that routes to the group).
"""

from __future__ import annotations

from .config import ParallelConfig
from .group import WorkerGroup, WorkerGroupError

__all__ = ["ParallelConfig", "WorkerGroup", "WorkerGroupError"]
