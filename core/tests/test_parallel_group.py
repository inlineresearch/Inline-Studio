from __future__ import annotations

from inline_core.device.policy import Parallel
from inline_core.parallel.config import ParallelConfig
from inline_core.parallel.group import WorkerGroup
from inline_core.parallel.launch import (
    DirectLauncher,
    TorchrunLauncher,
    default_launcher,
)


def _stub_config() -> ParallelConfig:
    return ParallelConfig(model="z-image-turbo", parallel=Parallel(), devices=(0,), stub=True)


def test_config_json_round_trip() -> None:
    config = ParallelConfig(model="z-image-turbo", parallel=Parallel(pipefusion=2), devices=(0, 1))
    restored = ParallelConfig.from_json(config.to_json())
    assert restored == config
    assert restored.world_size == 2


def test_default_launcher_selects_by_world_size() -> None:
    single = ParallelConfig(model="m", parallel=Parallel())
    multi = ParallelConfig(model="m", parallel=Parallel(pipefusion=2))
    assert isinstance(default_launcher(single), DirectLauncher)
    assert isinstance(default_launcher(multi), TorchrunLauncher)


def test_worker_group_round_trips_a_job() -> None:
    with WorkerGroup(_stub_config(), launcher=DirectLauncher()) as group:
        result = group.submit({"steps": 3, "note": "hello"})
    assert result["echo"]["note"] == "hello"
    assert result["model"] == "z-image-turbo"
    assert result["world_size"] == 1


def test_worker_group_streams_progress() -> None:
    seen: list[tuple[int, int]] = []
    with WorkerGroup(_stub_config(), launcher=DirectLauncher()) as group:
        group.submit({"steps": 4}, on_progress=lambda step, total: seen.append((step, total)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_worker_group_reuses_the_process_across_jobs() -> None:
    with WorkerGroup(_stub_config(), launcher=DirectLauncher()) as group:
        first = group.submit({"steps": 1, "n": 1})
        second = group.submit({"steps": 1, "n": 2})
    assert first["echo"]["n"] == 1
    assert second["echo"]["n"] == 2
