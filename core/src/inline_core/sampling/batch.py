"""The graph/GPU boundary. The graph never runs the denoise loop inline; it submits a SampleJob.

Phase 1 runs one job at a time behind this seam. Phase 5 adds cross-request batching (grouping by
model family, resolution, and adapter bucket) without changing the interface. Multi-GPU splits one
job across a worker group (XFuserBatchedSampler) when the device policy asks for a parallel denoise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..components.conditioning import Conditioning, Latents
from ..components.interfaces import Denoiser, Sampler, Scheduler, StepCallback, StepInfo

if TYPE_CHECKING:
    from ..device.policy import Placement
    from ..parallel.config import ParallelConfig
    from ..parallel.group import ProgressHandler
    from ..parallel.registry import GroupRegistry
    from ..runtime.context import ExecutionContext


@dataclass
class SampleJob:
    """One request to sample. Compatible jobs are grouped and stepped together."""

    denoiser: Denoiser
    scheduler: Scheduler
    sampler: Sampler
    latents: Latents
    conditioning: Conditioning
    steps: int
    on_step: StepCallback | None = None


class BatchedSampler(ABC):
    @abstractmethod
    def submit(self, job: SampleJob, ctx: ExecutionContext) -> Latents: ...


class InlineBatchedSampler(BatchedSampler):
    """Phase 1: no batching. Run the job through its sampler directly."""

    def submit(self, job: SampleJob, ctx: ExecutionContext) -> Latents:
        return job.sampler.sample(
            job.denoiser,
            job.scheduler,
            job.latents,
            job.conditioning,
            steps=job.steps,
            ctx=ctx,
            on_step=job.on_step,
        )


class ParallelCodec(Protocol):
    """Bridges a SampleJob and its worker group: which group, request out, latents back.

    The real codec moves torch tensors over a side channel and builds the xfuser request; it lives
    with the model runner (C2) so this module stays torch-free and the group stays mockable.
    """

    def config(self, placement: Placement) -> ParallelConfig:
        """The group identity (model + split) for this placement."""
        ...

    def to_request(self, job: SampleJob) -> dict[str, Any]:
        """The serializable request the worker runs against its resident pipeline."""
        ...

    def from_result(self, result: dict[str, Any], job: SampleJob) -> Latents:
        """The sampled latents the worker returned."""
        ...


class XFuserBatchedSampler(BatchedSampler):
    """Routes a parallel-placement denoise to the worker group; everything else runs inline.

    The device policy decides: a single-GPU or CPU run has no parallel placement, so it keeps the
    in-process path with zero overhead. Only a multi-GPU denoiser placement crosses to the group.
    """

    def __init__(
        self,
        registry: GroupRegistry,
        codec: ParallelCodec,
        inline: BatchedSampler | None = None,
    ) -> None:
        self._registry = registry
        self._codec = codec
        self._inline = inline or InlineBatchedSampler()

    def submit(self, job: SampleJob, ctx: ExecutionContext) -> Latents:
        placement = ctx.policy.placement("denoiser")
        if placement.parallel is None:
            return self._inline.submit(job, ctx)
        group = self._registry.get_or_create(self._codec.config(placement))
        result = group.submit(self._codec.to_request(job), _forward_progress(job))
        return self._codec.from_result(result, job)

    def close(self) -> None:
        self._registry.shutdown_all()


def _forward_progress(job: SampleJob) -> ProgressHandler | None:
    """Feed worker step ticks into the job's on_step callback, so streaming stays unchanged."""
    on_step = job.on_step
    if on_step is None:
        return None

    def report(step: int, total: int) -> None:
        on_step(StepInfo(step=step, total=total))

    return report
