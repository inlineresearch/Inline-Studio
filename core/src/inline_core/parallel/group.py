"""The manager side of a parallel denoise group: spawn, handshake, submit, shut down.

One group per parallel model config, spawned lazily on first parallel run and reused. The transport
is a TCP loopback socket (portable across macOS, Windows, Linux); rank 0 connects back to it. This
class stays torch-free so the seam is testable with the stub worker and no GPUs.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from ..errors import InlineCoreError
from .config import ADDR_ENV, CONFIG_ENV, ParallelConfig
from .launch import Launcher, default_launcher
from .protocol import MessageType, recv_message, send_message

ProgressHandler = Callable[[int, int], None]


class WorkerGroupError(InlineCoreError):
    """The worker group failed to start, died, or returned an error for a job."""


class WorkerGroup:
    def __init__(
        self,
        config: ParallelConfig,
        launcher: Launcher | None = None,
        *,
        ready_timeout: float = 120.0,
        shutdown_timeout: float = 10.0,
    ) -> None:
        self._config = config
        self._launcher = launcher or default_launcher(config)
        self._ready_timeout = ready_timeout
        self._shutdown_timeout = shutdown_timeout
        self._listener: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()
        self._listener = listener
        self._process = subprocess.Popen(
            self._launcher.command(self._config), env=self._child_env(host, port)
        )
        listener.settimeout(self._ready_timeout)
        try:
            conn, _ = listener.accept()
        except OSError as exc:
            self._terminate()
            raise WorkerGroupError("worker group did not connect in time") from exc
        conn.settimeout(None)
        self._conn = conn
        ready = recv_message(conn)
        if ready is None or ready.get("type") != MessageType.READY:
            self._terminate()
            raise WorkerGroupError("worker group did not report ready")

    def submit(
        self, payload: dict[str, Any], on_progress: ProgressHandler | None = None
    ) -> dict[str, Any]:
        conn = self._require_conn()
        send_message(conn, {"type": MessageType.JOB, "payload": payload})
        while True:
            message = recv_message(conn)
            if message is None:
                raise WorkerGroupError("worker group closed the connection mid-job")
            kind = message.get("type")
            if kind == MessageType.PROGRESS:
                if on_progress is not None:
                    on_progress(int(message["step"]), int(message["total"]))
            elif kind == MessageType.RESULT:
                result: dict[str, Any] = message["payload"]
                return result
            elif kind == MessageType.ERROR:
                raise WorkerGroupError(str(message.get("message", "worker group error")))
            else:
                raise WorkerGroupError(f"unexpected message type {kind!r}")

    def shutdown(self) -> None:
        if self._conn is not None:
            try:
                send_message(self._conn, {"type": MessageType.SHUTDOWN})
            except OSError:
                pass
        self._terminate()

    def _require_conn(self) -> socket.socket:
        if self._conn is None:
            raise WorkerGroupError("worker group is not started")
        return self._conn

    def _child_env(self, host: str, port: int) -> dict[str, str]:
        env = dict(os.environ)
        env[ADDR_ENV] = f"{host}:{port}"
        env[CONFIG_ENV] = self._config.to_json()
        # Let the worker import inline_core under an editable/src layout, where the package is on
        # sys.path but not on the inherited PYTHONPATH. Harmless once pip-installed.
        paths = [p for p in sys.path if p]
        existing = env.get("PYTHONPATH")
        if existing:
            paths = [existing, *paths]
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    def _terminate(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=self._shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self._process = None

    def __enter__(self) -> WorkerGroup:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
