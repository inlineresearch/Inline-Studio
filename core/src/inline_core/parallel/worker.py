"""Rank entrypoint for a parallel denoise group. Launched by a Launcher, one process per rank.

Rank 0 owns the IPC channel to the manager: it receives jobs, streams progress, and returns results
through a pluggable handler. The handler decides what a job means: the stub handler (no torch) backs
the scaffold and the round-trip test; the xfuser handler (loads the sharded pipeline and runs the
collective denoise across all ranks) lands with the Z-Image runner (C2).
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from typing import Any

from .config import ADDR_ENV, CONFIG_ENV, ParallelConfig
from .protocol import MessageType, recv_message, send_message

# report(step, total) streams progress to the manager while a job runs.
ProgressFn = Callable[[int, int], None]
# handler(config, payload, report) -> result payload.
Handler = Callable[[ParallelConfig, dict[str, Any], ProgressFn], dict[str, Any]]


def _stub_handler(
    config: ParallelConfig, payload: dict[str, Any], report: ProgressFn
) -> dict[str, Any]:
    total = int(payload.get("steps", 1))
    for step in range(1, total + 1):
        report(step, total)
    return {"echo": payload, "model": config.model, "world_size": config.world_size}


def _select_handler(config: ParallelConfig) -> Handler:
    if config.stub:
        return _stub_handler
    raise NotImplementedError(
        "the xfuser denoise handler lands with the Z-Image runner (C2); "
        "only the stub handler is available today"
    )


def serve(sock: socket.socket, config: ParallelConfig, handler: Handler) -> None:
    send_message(sock, {"type": MessageType.READY, "worldSize": config.world_size})
    while True:
        message = recv_message(sock)
        if message is None or message.get("type") == MessageType.SHUTDOWN:
            return
        if message.get("type") != MessageType.JOB:
            continue
        payload: dict[str, Any] = message.get("payload", {})

        def report(step: int, total: int) -> None:
            send_message(sock, {"type": MessageType.PROGRESS, "step": step, "total": total})

        try:
            result = handler(config, payload, report)
        except Exception as exc:  # translate into a protocol error; the manager re-raises typed
            send_message(sock, {"type": MessageType.ERROR, "message": str(exc)})
            continue
        send_message(sock, {"type": MessageType.RESULT, "payload": result})


def main() -> None:
    config = ParallelConfig.from_json(os.environ[CONFIG_ENV])
    host, _, port = os.environ[ADDR_ENV].partition(":")
    sock = socket.create_connection((host, int(port)))
    try:
        serve(sock, config, _select_handler(config))
    finally:
        sock.close()


if __name__ == "__main__":
    main()
