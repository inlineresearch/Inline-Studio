"""The Studio app-backend on Core: the browser SPA speaks the InlineStudioApi wire protocol —
``POST /rpc`` with ``{channel, args}`` plus an ``/events`` WebSocket, returning the same ``Result``
envelope (``{"ok": true, "value": ...}`` / ``{"ok": false, "error": ...}``). Core is the sole
backend; every channel is handled natively (the former proxy to the legacy Node backend is gone).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

#: A native channel handler: receives the positional args list, returns the value to wrap in Ok.
Handler = Callable[[list[Any]], Awaitable[Any]]


class RpcRouter:
    """Routes an RPC channel to its native handler."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, channel: str, handler: Handler) -> None:
        self._handlers[channel] = handler

    def has(self, channel: str) -> bool:
        return channel in self._handlers

    async def dispatch(self, channel: str, args: list[Any]) -> dict[str, Any]:
        handler = self._handlers.get(channel)
        if handler is None:
            return {"ok": False, "error": f"No handler registered for channel {channel!r}."}
        try:
            return {"ok": True, "value": await handler(args)}
        except Exception as error:  # noqa: BLE001 — Result envelope: errors never cross raw
            return {"ok": False, "error": str(error)}


class EventBroadcaster:
    """Fans server→client event frames out to every connected ``/events`` WebSocket subscriber.

    Native handlers push ``{"channel": ..., "payload": ...}`` frames here (matching the Node
    broadcaster's shape); each open socket drains its own queue. Bridging the legacy Node event
    stream is wired in with the fal domain (Part B4)."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def add(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def remove(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def broadcast(self, channel: str, payload: Any) -> None:
        frame = {"channel": channel, "payload": payload}
        for queue in self._subscribers:
            queue.put_nowait(frame)
