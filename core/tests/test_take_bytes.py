from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from helpers import FAKE_MODEL
from inline_core.graph.registry import build_default_registry
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import Node
from inline_core.runtime.context import ExecutionContext
from inline_core.runtime.file_store import FileTakeStore
from inline_core.server.app import create_app

_GRAPH = {
    "schemaVersion": 1,
    "nodes": [
        {"id": "p1", "type": "input/text", "params": {"text": "x"}},
        {"id": "m1", "type": "fake/model", "inputs": {"prompt": {"from": "p1", "output": "text"}}},
    ],
}


def test_take_bytes_served_from_disk(tmp_path: Path) -> None:
    takes = tmp_path / "takes"
    store = FileTakeStore(takes)

    class FileRunner(NodeRunner):
        produces_takes = True

        def run(
            self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext
        ) -> NodeResult:
            take = store.save(ctx.run_id, node.id, np.zeros((4, 4, 3), dtype=np.uint8), {})
            return NodeResult(outputs={"image": take}, takes=[take])

    registry = build_default_registry()
    registry.register(FAKE_MODEL, FileRunner())

    with TestClient(create_app(registry=registry, takes_dir=str(takes))) as client:
        run_id = client.post("/v1/runs", json={"graph": _GRAPH, "target": "m1"}).json()["runId"]
        state: dict[str, Any] = {}
        for _ in range(200):
            state = client.get(f"/v1/runs/{run_id}").json()
            if state["status"] == "done":
                break
            time.sleep(0.02)

        take_id = state["takes"][0]["id"]
        resp = client.get(f"/v1/takes/{take_id}/bytes")
        assert resp.status_code == 200
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert client.get("/v1/takes/bogus/bytes").status_code == 404
