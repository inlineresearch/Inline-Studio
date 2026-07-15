"""Fal relay: output parsing, API-key storage, and frame-input resolution to data URIs."""

from __future__ import annotations

from inline_core.studio import fal
from inline_core.studio import moodboard as mb
from inline_core.studio.store import StudioStore


def test_parse_image_and_video_outputs() -> None:
    resp = {"images": [{"url": "https://x/a.png", "content_type": "image/png"}]}
    img = fal.parse_outputs(resp, "image")
    assert img == [{"url": "https://x/a.png", "ext": ".png", "kind": "image"}]
    vid = fal.parse_outputs({"video": {"url": "https://x/clip.mp4"}}, "video")
    assert vid == [{"url": "https://x/clip.mp4", "ext": ".mp4", "kind": "video"}]
    # Empty / malformed shapes yield nothing.
    assert fal.parse_outputs({"images": []}, "image") == []
    assert fal.parse_outputs(None, "image") == []
    assert fal.parse_outputs({"images": [{"url": ""}]}, "image") == []


def test_ext_from_url_and_content_type() -> None:
    assert fal._ext_from("https://x/a.webp?token=1", None, ".png") == ".webp"
    assert fal._ext_from("https://x/noext", "image/jpeg", ".png") == ".jpg"
    assert fal._ext_from("https://x/noext", None, ".png") == ".png"


def test_fal_key_storage(tmp_path) -> None:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    assert store.fal_status() == {"hasKey": False, "encrypted": False}
    assert store.set_fal_key("  fal-secret  ")["hasKey"] is True
    assert store.fal_key() == "fal-secret"
    assert store.clear_fal_key()["hasKey"] is False
    assert store.fal_key() is None


def test_resolve_fal_inputs_media_and_prompt(tmp_path) -> None:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    project = store.create_project("Fal")
    conn, folder = store.conn(), store.folder()
    # An image asset on disk, wired as an input to a fal gen frame.
    (folder / "assets").mkdir(exist_ok=True)
    (folder / "assets" / "in.png").write_bytes(b"\x89PNG")
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('a1', ?, 'in', 'assets/in.png', 'image', 0)",
        (project["id"],),
    )
    gen = mb.add_gen_node(conn, "fal-ai/x", 0, 0, kind="image", params={}, title="X")
    conn.execute(
        "INSERT INTO frame_inputs (id, frame_id, asset_id, position) VALUES ('i1', ?, 'a1', 0)",
        (gen["frameId"],),
    )
    # A Prompt node wired into the gen node's 'prompt' handle.
    prompt = mb.add_prompt(conn, 0, 0)
    mb.update_item(conn, prompt["id"], {"data": {"promptText": "a fox"}})
    mb.create_connector(conn, prompt["id"], gen["id"], "out", "prompt")

    resolved = fal.resolve_fal_inputs(conn, folder, gen["frameId"])
    assert resolved["prompt"] == "a fox"
    assert len(resolved["images"]) == 1
    assert resolved["images"][0].startswith("data:image/png;base64,")
    assert resolved["videos"] == [] and resolved["audios"] == []
