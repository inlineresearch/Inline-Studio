"""The Studio /rpc bridge: native dispatch + the Result envelope (Core is the sole backend)."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from inline_core.server.app import create_app
from inline_core.server.rpc import RpcRouter


async def _echo(args: list) -> dict:
    return {"echo": args}


async def _boom(_args: list) -> None:
    raise ValueError("kaboom")


def test_native_handler_wraps_ok() -> None:
    router = RpcRouter()
    router.register("frames:list", _echo)
    assert asyncio.run(router.dispatch("frames:list", [1, 2])) == {
        "ok": True,
        "value": {"echo": [1, 2]},
    }


def test_handler_error_becomes_err_envelope() -> None:
    router = RpcRouter()
    router.register("frames:list", _boom)
    result = asyncio.run(router.dispatch("frames:list", []))
    assert result["ok"] is False and "kaboom" in result["error"]


def test_unknown_channel_is_err() -> None:
    result = asyncio.run(RpcRouter().dispatch("project:current", []))
    assert result["ok"] is False and "project:current" in result["error"]


def test_rpc_endpoint_dispatches() -> None:
    router = RpcRouter()
    router.register("app:version", _echo)
    app = create_app(rpc=router)
    with TestClient(app) as client:
        res = client.post("/rpc", json={"channel": "app:version", "args": ["x"]})
        assert res.json() == {"ok": True, "value": {"echo": ["x"]}}
        # A malformed body returns the Result envelope, not an HTTP error.
        bad = client.post("/rpc", json={"args": []})
        assert bad.json()["ok"] is False
