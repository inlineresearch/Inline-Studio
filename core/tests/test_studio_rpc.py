"""End-to-end: the Studio app-backend over Core's /rpc + /media + /upload (the B1 flip), driven
through the FastAPI app exactly as the browser SPA drives it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inline_core.device.memory import MemoryPolicy
from inline_core.graph.registry import build_default_registry
from inline_core.runtime.file_store import FileTakeStore
from inline_core.server.app import create_app
from inline_core.server.bootstrap import register_models
from inline_core.studio.store import StudioStore


@pytest.fixture
def client(tmp_path):
    store = StudioStore(tmp_path / "appdata", tmp_path / "workspace")
    registry = build_default_registry()
    register_models(registry, FileTakeStore(tmp_path / "takes"), MemoryPolicy())
    app = create_app(
        registry=registry,
        studio_store=store,
        asset_dir=str(tmp_path / "assets"),
        models_root=str(tmp_path / "models"),
        takes_dir=str(tmp_path / "takes"),
    )
    with TestClient(app) as c:
        yield c


def rpc(client, channel, *args):
    return client.post("/rpc", json={"channel": channel, "args": list(args)}).json()


def test_full_project_and_canvas_flow(client) -> None:
    # No project yet.
    assert rpc(client, "project:current")["value"] is None

    # Create → open state + recents.
    created = rpc(client, "project:create", {"name": "My Film", "parentDir": None})
    assert created["ok"] is True
    project = created["value"]
    assert project["name"] == "My Film"
    assert rpc(client, "project:current")["value"]["id"] == project["id"]
    assert rpc(client, "project:listRecent")["value"][0]["path"] == project["path"]

    # Canvas: add the Z-Image core node + a Prompt, then wire them.
    z = rpc(client, "moodboard:addCoreNode", "alibaba/z-image-turbo", 400, 200)["value"]
    assert z["type"] == "core" and z["data"]["core"]["type"] == "alibaba/z-image-turbo"
    prompt = rpc(client, "moodboard:addPrompt", 80, 200)["value"]
    conn = rpc(client, "moodboard:createConnector", prompt["id"], z["id"], "out", "prompt")
    assert conn["ok"] is True
    board = rpc(client, "moodboard:list")["value"]
    assert len(board["items"]) == 2 and len(board["connectors"]) == 1

    # Core node palette is served (Z-Image visible, primitives hidden).
    models = rpc(client, "core:models")["value"]
    visible = [m["type"] for m in models["models"] if not m.get("hidden")]
    assert visible == ["alibaba/z-image-turbo"]
    assert rpc(client, "core:status")["value"]["running"] is True

    # Settings round-trip.
    assert rpc(client, "settings:setCoreUrl", "http://127.0.0.1:9000")["value"]["coreUrl"] == (
        "http://127.0.0.1:9000"
    )


def test_asset_upload_and_media_serving(client) -> None:
    rpc(client, "project:create", {"name": "Media Proj", "parentDir": None})
    # Upload a file the way the browser does (POST /upload with raw bytes).
    up = client.post("/upload?name=pic.png", content=b"\x89PNG-bytes")
    assert up.json()["ok"] is True
    asset = up.json()["value"]
    assert asset["kind"] == "image"
    assert [a["id"] for a in rpc(client, "assets:list")["value"]] == [asset["id"]]

    # The file is now served over /media/<project-relative-path>.
    media = client.get("/media/" + asset["filePath"])
    assert media.status_code == 200
    assert media.content == b"\x89PNG-bytes"
    # Traversal is blocked.
    assert client.get("/media/../../etc/passwd").status_code in (403, 404)


def test_unported_channels_degrade_gracefully(client) -> None:
    rpc(client, "project:create", {"name": "P", "parentDir": None})
    # Embedded ComfyUI is desktop-only — a clear error, not a crash.
    cf = rpc(client, "comfy:linkFrame", "frame-x")
    assert cf["ok"] is False and "comfyui" in cf["error"].lower()
    # Cancel is a safe no-op.
    assert rpc(client, "generation:cancel")["ok"] is True


def test_fal_run_without_key_errors_via_event(client) -> None:
    rpc(client, "project:create", {"name": "P", "parentDir": None})
    # Fal run is fire-and-forget (ok immediately); the missing-key error arrives as an event.
    res = rpc(client, "generation:run", "frame-x", {"endpoint": "fal-ai/x", "body": {}})
    assert res["ok"] is True
