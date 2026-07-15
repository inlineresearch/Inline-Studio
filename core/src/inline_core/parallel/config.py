"""What a parallel denoise group is: which model, split how, on which CUDA devices.

The config crosses a process boundary (env var -> worker), so it is JSON-portable. `stub` selects a
torch-free handler used by the scaffold and the IPC round-trip test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..device.policy import Parallel

# Env vars the manager sets on the worker process (read in worker.main).
ADDR_ENV = "INLINE_PARALLEL_ADDR"
CONFIG_ENV = "INLINE_PARALLEL_CONFIG"


@dataclass(frozen=True)
class ParallelConfig:
    """Identity of a parallel group. One group per (model, parallel, devices) combination."""

    model: str
    parallel: Parallel = field(default_factory=Parallel)
    devices: tuple[int, ...] = ()
    stub: bool = False

    @property
    def world_size(self) -> int:
        return self.parallel.world_size

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": self.model,
                "parallel": asdict(self.parallel),
                "devices": list(self.devices),
                "stub": self.stub,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> ParallelConfig:
        data = json.loads(raw)
        return cls(
            model=data["model"],
            parallel=Parallel(**data["parallel"]),
            devices=tuple(data["devices"]),
            stub=bool(data.get("stub", False)),
        )
