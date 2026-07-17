"""Framed JSON messages over a stream socket. Length-prefixed so reads never split a message.

Control only: jobs, progress, results, shutdown. Large latent/image tensors move over a side channel
(shared memory) added with the xfuser handler; the control plane stays small and JSON-portable.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

_HEADER = struct.Struct(">I")  # 4-byte big-endian body length


class MessageType:
    READY = "ready"  # worker -> manager: connected and serving
    JOB = "job"  # manager -> worker: run this payload
    PROGRESS = "progress"  # worker -> manager: step of total
    RESULT = "result"  # worker -> manager: job payload result
    ERROR = "error"  # worker -> manager: job failed, message attached
    SHUTDOWN = "shutdown"  # manager -> worker: stop serving and exit


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    body = json.dumps(message).encode("utf-8")
    sock.sendall(_HEADER.pack(len(body)) + body)


def recv_message(sock: socket.socket) -> dict[str, Any] | None:
    """The next message, or None when the peer closed the connection cleanly."""
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    body = _recv_exactly(sock, length)
    if body is None:
        return None
    decoded: dict[str, Any] = json.loads(body.decode("utf-8"))
    return decoded


def _recv_exactly(sock: socket.socket, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
