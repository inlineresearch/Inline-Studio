"""Periodic host + GPU telemetry, broadcast on ``events:systemStats`` for the Trainer tab.

Samples CPU/RAM (psutil) and per-GPU utilization/VRAM/temp (pynvml) about once a second and fans it
out to every ``/events`` subscriber. Skips sampling entirely when nobody is listening, so it costs
nothing with no client connected. GPU stats are NVIDIA-only; an MPS/CPU host reports host metrics
with an empty ``gpus`` list. Missing psutil/pynvml degrade to zeros rather than erroring.
"""

from __future__ import annotations

import asyncio
from typing import Any

_INTERVAL_SECONDS = 1.5


class SystemStats:
    """A background poller that broadcasts ``SystemStatsEvent`` frames while clients connect."""

    def __init__(self, events: Any) -> None:
        self._events = events
        self._task: asyncio.Task[None] | None = None
        self._nvml_ready = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_INTERVAL_SECONDS)
                if self._events.subscriber_count == 0:
                    continue  # nobody watching - don't spend a sample
                stats = await asyncio.to_thread(self._sample)
                self._events.broadcast("events:systemStats", stats)
        except asyncio.CancelledError:
            pass

    def _sample(self) -> dict[str, Any]:
        cpu, ram_used, ram_total = self._host()
        return {"cpu": cpu, "ramUsed": ram_used, "ramTotal": ram_total, "gpus": self._gpus()}

    def _host(self) -> tuple[float, int, int]:
        try:
            import psutil

            vm = psutil.virtual_memory()
            return float(psutil.cpu_percent()), int(vm.used), int(vm.total)
        except Exception:  # noqa: BLE001 - psutil optional; telemetry degrades to zeros
            return 0.0, 0, 0

    def _gpus(self) -> list[dict[str, Any]]:
        try:
            import pynvml
        except Exception:  # noqa: BLE001 - no NVML (non-NVIDIA or extra not installed)
            return []
        try:
            if not self._nvml_ready:
                pynvml.nvmlInit()
                self._nvml_ready = True
            out: list[dict[str, Any]] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                out.append(
                    {
                        "index": i,
                        "name": name.decode() if isinstance(name, bytes) else name,
                        "utilization": float(util.gpu),
                        "memoryUsed": int(mem.used),
                        "memoryTotal": int(mem.total),
                        "temperature": _temperature(pynvml, handle),
                    }
                )
            return out
        except Exception:  # noqa: BLE001 - any NVML hiccup: report no GPUs rather than crash
            return []


def _temperature(pynvml: Any, handle: Any) -> float | None:
    try:
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:  # noqa: BLE001
        return None
