"""End-to-end: the character library over /rpc, /upload/character and /download/character,
driven through the FastAPI app exactly as the browser SPA drives it."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inline_core.device.memory import MemoryPolicy
from inline_core.graph.registry import build_default_registry
from inline_core.runtime.file_store import FileTakeStore
from inline_core.server.app import create_app
from inline_core.server.bootstrap import register_models
from inline_core.studio.store import StudioStore

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models_root = tmp_path / "models"
    monkeypatch.setenv("INLINE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)
    store = StudioStore(tmp_path / "appdata", tmp_path / "workspace")
    registry = build_default_registry()
    register_models(registry, FileTakeStore(tmp_path / "takes"), MemoryPolicy())
    app = create_app(
        registry=registry,
        studio_store=store,
        asset_dir=str(tmp_path / "assets"),
        models_root=str(models_root),
        takes_dir=str(tmp_path / "takes"),
    )
    with TestClient(app) as c:
        yield c


def rpc(client: TestClient, channel: str, *args: object) -> dict:
    return client.post("/rpc", json={"channel": channel, "args": list(args)}).json()


def _upload_image(client: TestClient, name: str, colour: tuple[int, int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), colour).save(buffer, format="PNG")
    response = client.post(
        f"/upload?name={name}", content=buffer.getvalue()
    ).json()
    assert response["ok"] is True, response
    return str(response["value"]["id"])


@pytest.fixture
def project(client: TestClient) -> dict:
    created = rpc(client, "project:create", {"name": "Char Film", "parentDir": None})
    assert created["ok"] is True
    return created["value"]


def test_the_library_starts_empty(client: TestClient, project: dict) -> None:
    assert rpc(client, "characters:list") == {"ok": True, "value": []}


def test_create_list_edit_and_delete(client: TestClient, project: dict) -> None:
    asset = _upload_image(client, "ada.png", (120, 90, 60))

    created = rpc(
        client,
        "characters:create",
        {"name": "Ada", "assetIds": [asset], "description": "green canvas jacket"},
    )
    assert created["ok"] is True, created
    summary = created["value"]
    assert summary["name"] == "Ada"
    assert summary["file"] == "Ada.char"
    assert summary["refs"] == 1
    assert summary["description"] == "green canvas jacket"
    # Creating from one image is allowed, and the nudge to add more is advice rather than an error.
    assert summary["hints"] == ["Add a second angle"]

    listed = rpc(client, "characters:list")["value"]
    assert [row["file"] for row in listed] == ["Ada.char"]

    detail = rpc(client, "characters:get", "Ada.char")["value"]
    assert detail["refUrls"] == ["/character-ref/Ada.char/0"]

    renamed = rpc(client, "characters:rename", "Ada.char", "Ada Lovelace")["value"]
    assert renamed["name"] == "Ada Lovelace"
    # The filename is the dropdown's stored value, so a rename must not move it.
    assert renamed["file"] == "Ada.char"

    described = rpc(client, "characters:setDescription", "Ada.char", "red scarf")["value"]
    assert described["description"] == "red scarf"

    assert rpc(client, "characters:delete", "Ada.char")["value"] is True
    assert rpc(client, "characters:list")["value"] == []


def test_create_is_refused_without_a_name_or_a_reference(
    client: TestClient, project: dict
) -> None:
    asset = _upload_image(client, "a.png", (10, 20, 30))
    blank = rpc(client, "characters:create", {"name": "", "assetIds": [asset]})
    assert blank["ok"] is False and "name" in blank["error"]

    empty = rpc(client, "characters:create", {"name": "Ada", "assetIds": []})
    assert empty["ok"] is False and "reference" in empty["error"]


def test_refs_can_be_added_and_removed_but_never_all_of_them(
    client: TestClient, project: dict
) -> None:
    first = _upload_image(client, "a.png", (10, 20, 30))
    second = _upload_image(client, "b.png", (200, 40, 40))
    rpc(client, "characters:create", {"name": "Ada", "assetIds": [first]})

    added = rpc(client, "characters:addRefs", "Ada.char", [second])
    assert added["ok"] is True and added["value"]["refs"] == 2
    # The identity survives a rebuild, so takes that recorded it still point at this character.
    assert added["value"]["charId"] == rpc(client, "characters:get", "Ada.char")["value"]["charId"]

    removed = rpc(client, "characters:removeRef", "Ada.char", 0)
    assert removed["ok"] is True and removed["value"]["refs"] == 1

    last = rpc(client, "characters:removeRef", "Ada.char", 0)
    assert last["ok"] is False and "at least one" in last["error"]


def test_a_reference_thumbnail_is_served_without_unzipping_in_the_browser(
    client: TestClient, project: dict
) -> None:
    asset = _upload_image(client, "a.png", (10, 20, 30))
    rpc(client, "characters:create", {"name": "Ada", "assetIds": [asset]})

    response = client.get("/character-ref/Ada.char/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (640, 480)

    assert client.get("/character-ref/Ada.char/9").status_code == 404
    assert client.get("/character-ref/Nobody.char/0").status_code == 404


def test_export_then_import_round_trips_through_the_routes(
    client: TestClient, project: dict
) -> None:
    asset = _upload_image(client, "a.png", (10, 20, 30))
    rpc(client, "characters:create", {"name": "Ada", "assetIds": [asset]})

    exported = client.get("/download/character/Ada.char")
    assert exported.status_code == 200
    assert exported.headers["content-type"] == "application/octet-stream"

    rpc(client, "characters:delete", "Ada.char")
    assert rpc(client, "characters:list")["value"] == []

    imported = client.post(
        "/upload/character?name=Ada.char", content=exported.content
    ).json()
    assert imported["ok"] is True
    assert imported["value"]["file"] == "Ada.char"
    assert rpc(client, "characters:list")["value"][0]["name"] == "Ada"


def test_importing_something_that_is_not_a_character_is_refused(
    client: TestClient, project: dict
) -> None:
    response = client.post("/upload/character?name=evil.char", content=b"not a zip").json()
    assert response["ok"] is False
    assert rpc(client, "characters:list")["value"] == []


def test_download_is_confined_to_the_characters_folder(
    client: TestClient, project: dict
) -> None:
    """The name arrives from the browser, so a traversal attempt must not reach the models root."""
    assert client.get("/download/character/..%2F..%2Fsecret.txt").status_code == 404


def test_a_character_appears_in_the_flux2_node_dropdown(
    client: TestClient, project: dict
) -> None:
    """The whole point of the models-root location: the node picker fills itself."""
    asset = _upload_image(client, "a.png", (10, 20, 30))
    rpc(client, "characters:create", {"name": "Ada", "assetIds": [asset]})

    models = rpc(client, "core:models")["value"]["models"]
    flux2 = next((m for m in models if m["type"] == "black-forest-labs/flux-2"), None)
    if flux2 is None:
        pytest.skip("FLUX.2 is not registered on this install")
    field = next(p for p in flux2["params"] if p["key"] == "character")
    assert "Ada.char" in [o["value"] for o in field["options"]]


def test_a_character_written_before_the_current_encoders_rebuilds_on_open(
    client: TestClient, project: dict, tmp_path: Path
) -> None:
    """Refs are truth and scoring is cache. Without this, a character written before the current
    encoders keeps a centroid the version check correctly rejects and nothing ever restores, and
    never gains the per-reference framings the full-body hint and the subject term both need."""
    from inline_core.characters import charfile as cf
    from inline_core.characters import library

    asset = _upload_image(client, "ada.png", (120, 90, 60))
    rpc(client, "characters:create", {"name": "Ada", "assetIds": [asset]})
    path = library.resolve("Ada.char")
    assert path is not None

    aged = cf.read(path)
    aged.manifest.scoring = {
        **aged.manifest.scoring,
        "refFramings": [],
        "encoders": [{"id": "dinov2-base", "version": "0", "dim": 768}],
    }
    cf.write(path, aged)

    rpc(client, "characters:get", "Ada.char")

    rebuilt = cf.read(path)
    assert not cf.centroid_valid(rebuilt.manifest, "dinov2-base", "0"), "stale version survived"
    assert "refFramings" in rebuilt.manifest.scoring, "framings were not recomputed"


def test_creating_a_character_streams_its_encode_phases(
    client: TestClient, project: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phases arrive while the create call is still open, which is what makes them useful."""
    from inline_core.server.rpc import EventBroadcaster

    seen: list[tuple[str, object]] = []
    original = EventBroadcaster.broadcast

    def record(self: EventBroadcaster, channel: str, payload: object) -> None:
        seen.append((channel, payload))
        original(self, channel, payload)

    monkeypatch.setattr(EventBroadcaster, "broadcast", record)

    asset = _upload_image(client, "ada.png", (120, 90, 60))
    created = rpc(client, "characters:create", {"name": "Ada", "assetIds": [asset]})
    assert created["ok"] is True, created

    phases = [p for channel, p in seen if channel == "events:characterProgress"]
    assert phases, f"no progress was broadcast; saw {[c for c, _ in seen]}"
    assert all(isinstance(p, dict) and p["name"] == "Ada" for p in phases)
    fractions = [p["fraction"] for p in phases]  # type: ignore[index]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    # The library-changed push still fires, so the list refreshes as well as the bar.
    assert any(channel == "events:charactersChanged" for channel, _ in seen)
