from __future__ import annotations

from typing import Any

from inline_core.components.conditioning import Conditioning, Latents
from inline_core.device.memory import MemoryPolicy
from inline_core.device.policy import Parallel, Placement
from inline_core.device.types import Device, DeviceKind
from inline_core.parallel.config import ParallelConfig
from inline_core.parallel.registry import GroupRegistry
from inline_core.runtime.context import CancelToken, ExecutionContext
from inline_core.runtime.progress import ProgressEmitter, RunEvent
from inline_core.sampling.batch import BatchedSampler, SampleJob, XFuserBatchedSampler


class _NullEmitter(ProgressEmitter):
    def emit(self, event: RunEvent) -> None:
        pass


class _RecordingInline(BatchedSampler):
    def __init__(self) -> None:
        self.calls = 0

    def submit(self, job: SampleJob, ctx: ExecutionContext) -> Latents:
        self.calls += 1
        return Latents(tensor="inline")  # type: ignore[arg-type]


class _StubCodec:
    """Routes to a single-process stub worker and echoes its payload back as the latents."""

    def config(self, placement: Placement) -> ParallelConfig:
        return ParallelConfig(model="z-image-turbo", parallel=Parallel(), devices=(0,), stub=True)

    def to_request(self, job: SampleJob) -> dict[str, Any]:
        return {"steps": job.steps}

    def from_result(self, result: dict[str, Any], job: SampleJob) -> Latents:
        return Latents(tensor=result)  # type: ignore[arg-type]


def _cuda(count: int) -> tuple[Device, ...]:
    return tuple(Device(DeviceKind.CUDA, i) for i in range(count))


def _ctx(policy: MemoryPolicy) -> ExecutionContext:
    return ExecutionContext(run_id="r", policy=policy, emitter=_NullEmitter(), cancel=CancelToken())


def _job(steps: int, on_step: Any = None) -> SampleJob:
    stand_in: Any = object()
    return SampleJob(
        stand_in, stand_in, stand_in, Latents(tensor="x"), Conditioning(), steps, on_step
    )  # type: ignore[arg-type]


def test_single_gpu_placement_runs_inline() -> None:
    inline = _RecordingInline()
    sampler = XFuserBatchedSampler(GroupRegistry(), _StubCodec(), inline=inline)
    result = sampler.submit(_job(steps=5), _ctx(MemoryPolicy(devices=_cuda(1), vram_gb=24)))
    sampler.close()
    assert inline.calls == 1
    assert result.tensor == "inline"


def test_multi_gpu_placement_routes_to_worker_group() -> None:
    seen: list[tuple[int, int]] = []
    inline = _RecordingInline()
    sampler = XFuserBatchedSampler(GroupRegistry(), _StubCodec(), inline=inline)
    policy = MemoryPolicy(devices=_cuda(2), vram_gb=24, nvlink=False)
    job = _job(steps=3, on_step=lambda info: seen.append((info.step, info.total)))
    result = sampler.submit(job, _ctx(policy))
    sampler.close()
    assert inline.calls == 0
    assert result.tensor["echo"]["steps"] == 3
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_registry_reuses_one_group_per_config() -> None:
    registry = GroupRegistry()
    config = ParallelConfig(model="z-image-turbo", parallel=Parallel(), devices=(0,), stub=True)
    try:
        first = registry.get_or_create(config)
        second = registry.get_or_create(config)
        assert first is second
    finally:
        registry.shutdown_all()
