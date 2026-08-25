"""End-to-end: the character library over /rpc, /upload/character and /download/character, driven
through the FastAPI app exactly as the browser SPA drives it.

Creating, editing and building a character are the character nodes' job now, so those live in
tests/test_character_nodes.py; what is left here is the browser-facing library surface.
"""

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


@pytest.fixture(scope="session")
def _encoder_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One copy of the ~385MB scoring weights for the whole session, not one per test."""
    root = tmp_path_factory.mktemp("encoders")
    import os

    from inline_core.characters import weights

    previous = os.environ.get("INLINE_MODELS_DIR")
    os.environ["INLINE_MODELS_DIR"] = str(root)
    try:
        weights.ensure()
    except Exception as error:  # noqa: BLE001 - offline is a skip, not a failure
        pytest.skip(f"character encoders unavailable: {error}")
    finally:
        if previous is None:
            os.environ.pop("INLINE_MODELS_DIR", None)
        else:
            os.environ["INLINE_MODELS_DIR"] = previous
    return root / "annotators"


@pytest.fixture(autouse=True)
def _encoders(client: TestClient, tmp_path: Path, _encoder_root: Path) -> None:
    """Point this test's models root at the shared weights; encoding refuses without them."""
    link = tmp_path / "models" / "annotators"
    link.parent.mkdir(parents=True, exist_ok=True)
    # The catalog creates every category folder on boot, so the empty one is replaced here.
    if link.is_dir() and not link.is_symlink():
        link.rmdir()
    if not link.is_symlink():
        link.symlink_to(_encoder_root, target_is_directory=True)


class _NullEvents:
    def broadcast(self, *_args: object) -> None:
        return None


def _make_character(name: str = "Ada", description: str = "green canvas jacket") -> str:
    """One character, made the way the nodes make one: Encode then Write. Returns its filename.

    Driven through the runners rather than the library directly, so the save listener that keeps
    the node picker current is exercised too."""
    import tempfile

    from inline_core.graph.schema import Node
    from inline_core.models.character.runner import EncodeCharacterRunner, WriteCharacterRunner
    from inline_core.runtime.context import CancelToken, ExecutionContext
    from inline_core.runtime.progress import NullEmitter
    from inline_core.takes import AssetRef

    folder = Path(tempfile.mkdtemp(prefix="char-fixture-"))
    ref = folder / "a.png"
    Image.new("RGB", (640, 480), (120, 90, 60)).save(ref)
    ctx = ExecutionContext(
        run_id="r", policy=object(), emitter=NullEmitter(), cancel=CancelToken()  # type: ignore[arg-type]
    )
    node = Node(id="n", type="character/encode", params={"name": name, "description": description})
    identity = EncodeCharacterRunner().run(
        node, {"images": [AssetRef(ref=str(ref), path=str(ref))]}, ctx
    ).outputs["character"]
    written = WriteCharacterRunner().run(
        Node(id="w", type="character/write", params={}),
        {"character": [identity], "payloads": []},
        ctx,
    ).outputs["character"]
    return str(written.file)


@pytest.fixture
def project(client: TestClient) -> dict:
    created = rpc(client, "project:create", {"name": "Char Film", "parentDir": None})
    assert created["ok"] is True
    return created["value"]


def test_the_library_starts_empty(client: TestClient, project: dict) -> None:
    assert rpc(client, "characters:list") == {"ok": True, "value": []}


def test_a_reference_thumbnail_is_served_without_unzipping_in_the_browser(
    client: TestClient, project: dict
) -> None:
    _make_character()

    response = client.get("/character-ref/Ada.char/0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (640, 480)

    assert client.get("/character-ref/Ada.char/9").status_code == 404
    assert client.get("/character-ref/Nobody.char/0").status_code == 404


def test_export_then_import_round_trips_through_the_routes(
    client: TestClient, project: dict
) -> None:
    _make_character()

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


def test_a_character_appears_in_the_load_node_picker(client: TestClient, project: dict) -> None:
    """The whole point of the models-root location: the node picker fills itself. A generation node
    takes its character by wire, so the picker that names one is Load Character."""
    _make_character()

    models = rpc(client, "core:models")["value"]["models"]
    load = next((m for m in models if m["type"] == "character/load"), None)
    if load is None:
        pytest.skip("character nodes are not registered on this install")
    field = next(p for p in load["params"] if p["key"] == "file")
    assert "Ada.char" in [o["value"] for o in field["options"]]


def test_a_character_written_before_the_current_encoders_rebuilds_when_applied(
    client: TestClient, project: dict, tmp_path: Path
) -> None:
    """Refs are truth and scoring is cache. Without this, a character written before the current
    encoders keeps a centroid the version check correctly rejects and nothing ever restores, and
    never gains the per-reference framings the full-body hint and the subject term both need.

    Applying is where it matters, and where it now happens: there is no editor to open."""
    from inline_core.characters import charfile as cf
    from inline_core.characters import library

    _make_character()
    path = library.resolve("Ada.char")
    assert path is not None

    aged = cf.read(path)
    aged.manifest.scoring = {
        **aged.manifest.scoring,
        "refFramings": [],
        "encoders": [{"id": "dinov2-base", "version": "0", "dim": 768}],
    }
    cf.write(path, aged)

    # Scoring a take is where a stale centroid bites, so it is where the rebuild happens.
    from inline_core.studio.characters import Characters

    Characters(object(), _NullEvents()).score_take(path, "Ada.char")

    rebuilt = cf.read(path)
    assert not cf.centroid_valid(rebuilt.manifest, "dinov2-base", "0"), "stale version survived"
    assert "refFramings" in rebuilt.manifest.scoring, "framings were not recomputed"


def test_rescoring_a_stale_character_keeps_its_trained_adapter(
    client: TestClient, project: dict
) -> None:
    """Rebuilding scoring must not rebuild the character.

    `char_encode` builds a fresh manifest and a fresh members dict, so rescoring through it dropped
    the trained adapter, every payload but flux2-klein and the apply override - and the write that
    follows put that on disk, from a path a render reaches on every take it scores.
    """
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode, library
    from inline_core.studio.characters import Characters

    _make_character()
    path = library.resolve("Ada.char")
    assert path is not None

    doc = cf.read(path)
    encode.set_lora_payload(
        doc.manifest,
        doc.members,
        b"adapter-bytes",
        arch=encode.FLUX2_KLEIN_ARCH,
        base="flux2-klein-4b",
        rank=16,
        steps=500,
        resolution=512,
    )
    doc.manifest.payloads["minimax-h3"] = {"payload_version": 1, "type": "ref", "files": []}
    doc.manifest.apply[encode.FLUX2_KLEIN_ARCH] = "lora"
    doc.manifest.reserved = {"adapters": {}, "video_payloads": {}, "members": ["keep-me"]}
    # The state a shipped encoder bump puts every character on disk into.
    doc.manifest.scoring = {
        **doc.manifest.scoring,
        "encoders": [{"id": "dinov2-base", "version": "0", "dim": 768}],
    }
    cf.write(path, doc)

    Characters(object(), _NullEvents()).score_take(path, "Ada.char")

    rebuilt = cf.read(path)
    key = encode.payload_key(encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_LORA)
    assert key in rebuilt.manifest.payloads, "the trained adapter was destroyed"
    assert rebuilt.members[f"payloads/{key}/adapter.safetensors"] == b"adapter-bytes"
    assert "minimax-h3" in rebuilt.manifest.payloads, "another model's payload was destroyed"
    assert rebuilt.manifest.apply.get(encode.FLUX2_KLEIN_ARCH) == "lora"
    assert rebuilt.manifest.reserved.get("members") == ["keep-me"]
    # And it did actually rescore, or the assertions above pass on a file nothing touched.
    assert cf.centroid_valid(rebuilt.manifest, "dinov2-base", "2"), "scoring was not rebuilt"
