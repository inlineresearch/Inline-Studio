from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient
from helpers import FAKE_MODEL
from inline_core.graph.cache import InMemoryCache
from inline_core.graph.registry import build_default_registry
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import Node
from inline_core.media import MediaKind
from inline_core.runtime.context import ExecutionContext
from inline_core.runtime.progress import Phase, ProgressEvent
from inline_core.server.app import create_app
from inline_core.takes import Take

_GRAPH = {
    "schemaVersion": 1,
    "nodes": [
        {"id": "p1", "type": "input/text", "params": {"text": "a fox"}},
        {
            "id": "m1",
            "type": "fake/model",
            "params": {"seed": 7},
            "inputs": {"prompt": {"from": "p1", "output": "text"}},
        },
    ],
}


class ServerFakeRunner(NodeRunner):
    """Sleeps briefly so a websocket client reliably sees the run in flight."""

    produces_takes = True

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        ctx.emitter.emit(
            ProgressEvent(ctx.run_id, node.id, Phase.SAMPLE, 0.5, step=1, step_count=2)
        )
        time.sleep(0.05)
        take = Take(
            id=f"take-{node.id}",
            run_id=ctx.run_id,
            node_id=node.id,
            kind=MediaKind.IMAGE,
            uri=f"mem://{node.id}",
            hash=f"h-{node.id}",
            params=dict(node.params),
        )
        return NodeResult(outputs={"image": take}, takes=[take])


def _client() -> TestClient:
    registry = build_default_registry()
    registry.register(FAKE_MODEL, ServerFakeRunner())
    return TestClient(create_app(registry=registry, cache=InMemoryCache()))


def _wait_done(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        state = client.get(f"/v1/runs/{run_id}").json()
        if state["status"] in ("done", "error", "cancelled"):
            return state
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_health_and_models() -> None:
    with _client() as client:
        health = client.get("/v1/health").json()
        assert health["ok"] and health["apiVersion"] == "v1"
        assert "registryVersion" in health
        models = client.get("/v1/models").json()
        assert models["registryVersion"].startswith("r_")
        types = {m["type"] for m in models["models"]}
        assert "fake/model" in types
        assert "input/text" in types


def test_models_etag_304() -> None:
    with _client() as client:
        etag = client.get("/v1/models").headers["etag"]
        assert client.get("/v1/models", headers={"If-None-Match": etag}).status_code == 304


def test_run_lifecycle_poll() -> None:
    with _client() as client:
        resp = client.post("/v1/runs", json={"graph": _GRAPH, "target": "m1"})
        assert resp.status_code == 201
        state = _wait_done(client, resp.json()["runId"])
        assert state["status"] == "done"
        assert state["fraction"] == 1.0
        assert state["nodes"]["m1"]["state"] == "done"
        assert len(state["takes"]) == 1


def test_invalid_graph_returns_422() -> None:
    with _client() as client:
        bad = {"schemaVersion": 1, "nodes": [{"id": "m1", "type": "fake/model"}]}
        resp = client.post("/v1/runs", json={"graph": bad, "target": "m1"})
        assert resp.status_code == 422
        assert resp.json()["error"]["nodeId"] == "m1"


def test_websocket_streams_events() -> None:
    with _client() as client:
        run_id = client.post("/v1/runs", json={"graph": _GRAPH, "target": "m1"}).json()["runId"]
        with client.websocket_connect(f"/v1/runs/{run_id}/events") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot"
            if first["state"]["status"] == "done":
                return
            seen: list[str] = []
            for _ in range(50):
                message = ws.receive_json()
                seen.append(message["type"])
                if message["type"] == "run_done":
                    break
            assert "run_done" in seen
            assert "node_done" in seen
