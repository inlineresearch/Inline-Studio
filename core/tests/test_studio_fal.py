"""Fal relay: output parsing, API-key storage, and frame-input resolution to data URIs."""

from __future__ import annotations

import pytest

from inline_core.studio import fal
from inline_core.studio import frames as fr
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


def test_parse_audio_outputs() -> None:
    """Audio is the shape the Sonilo music nodes return, and was previously uncovered here - the
    only tests for it lived in TypeScript against a `parseOutputs` the engine never called."""
    single = fal.parse_outputs({"audio": {"url": "https://x/t.m4a", "content_type": "audio/mp4"}},
                               "audio")
    assert single == [{"url": "https://x/t.m4a", "ext": ".m4a", "kind": "audio"}]
    # The plural array is used when there is no singular slot (e.g. multi-sample responses).
    many = fal.parse_outputs(
        {"audios": [{"url": "https://x/a.m4a"}, {"url": "https://x/b.m4a"}]}, "audio"
    )
    assert [r["url"] for r in many] == ["https://x/a.m4a", "https://x/b.m4a"]
    assert fal.parse_outputs({"audios": []}, "audio") == []
    assert fal.parse_outputs({"audio": {"url": ""}}, "audio") == []
    assert fal.parse_outputs(None, "audio") == []


def test_parse_outputs_prefers_the_singular_slot_over_the_array() -> None:
    """Documents current behaviour: when fal returns BOTH `audio` and `audios`, only the singular
    slot is taken - so a `num_samples > 1` request yields one take. Captured here so the trade-off
    is visible and any future change to multi-sample handling is a deliberate, test-breaking one."""
    resp = {
        "audio": {"url": "https://x/first.m4a"},
        "audios": [{"url": "https://x/first.m4a"}, {"url": "https://x/second.m4a"}],
    }
    assert [r["url"] for r in fal.parse_outputs(resp, "audio")] == ["https://x/first.m4a"]


def test_parse_video_outputs_singular_and_plural() -> None:
    plural = fal.parse_outputs({"videos": [{"url": "https://x/a.mp4"}]}, "video")
    assert plural == [{"url": "https://x/a.mp4", "ext": ".mp4", "kind": "video"}]
    assert fal.parse_outputs({"video": {"url": ""}}, "video") == []
    assert fal.parse_outputs({"video": "not-a-dict"}, "video") == []


def test_ext_from_url_and_content_type() -> None:
    assert fal._ext_from("https://x/a.webp?token=1", None, ".png") == ".webp"
    assert fal._ext_from("https://x/noext", "image/jpeg", ".png") == ".jpg"
    assert fal._ext_from("https://x/noext", None, ".png") == ".png"


def test_ext_disambiguates_audio_mp4_from_video_mp4() -> None:
    """`audio/mp4` and `video/mp4` share the subtype "mp4"; mapping on the subtype alone saved
    Sonilo's audio takes as .mp4. The extension must follow the full content type."""
    assert fal._ext_from("https://x/t", "audio/mp4", ".mp3") == ".m4a"
    assert fal._ext_from("https://x/t", "video/mp4", ".mp4") == ".mp4"
    assert fal._ext_from("https://x/t", "audio/mpeg", ".mp3") == ".mp3"
    # A charset/parameter suffix must not defeat the lookup.
    assert fal._ext_from("https://x/t", "audio/mp4; codecs=mp4a.40.2", ".mp3") == ".m4a"
    # Unlisted types still fall back to the subtype.
    assert fal._ext_from("https://x/t", "image/avif", ".png") == ".avif"


class _FakeResponse:
    """Stands in for an httpx response on an error path."""

    def __init__(self, status_code: int, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeHTTPError(Exception):
    def __init__(self, response: _FakeResponse) -> None:
        super().__init__("Client error '422 Unprocessable Entity' for url 'https://queue.fal.run/x'")
        self.response = response


def test_fal_error_message_names_a_content_filter_rejection() -> None:
    """The reported case: a 422 whose body explains the prompt was refused by a safety filter.
    Previously the user only saw httpx's generic text + an MDN link, with fal's reason discarded."""
    detail = "The content could not be processed because it contained material flagged by a " \
             "content moderation policy."
    msg = fal.fal_error_message(_FakeHTTPError(_FakeResponse(422, {"detail": detail})))
    assert "content filter" in msg
    assert "nothing was generated" in msg
    assert detail in msg  # the provider's own wording is preserved
    assert "Adjust the prompt or input image" in msg
    assert "MDN" not in msg and "developer.mozilla" not in msg


def test_fal_error_message_maps_common_statuses() -> None:
    auth = fal.fal_error_message(_FakeHTTPError(_FakeResponse(401, {"detail": "bad key"})))
    assert "API key" in auth and "Settings" in auth
    rate = fal.fal_error_message(_FakeHTTPError(_FakeResponse(429, {"detail": "slow down"})))
    assert "rate-limiting" in rate
    missing = fal.fal_error_message(_FakeHTTPError(_FakeResponse(404, {"detail": "nope"})))
    assert "no such model endpoint" in missing
    server = fal.fal_error_message(_FakeHTTPError(_FakeResponse(503, {"detail": "down"})))
    assert "server error" in server


def test_fal_error_detail_handles_falsy_and_odd_shapes() -> None:
    # FastAPI-style validation list.
    validation = fal.fal_error_message(
        _FakeHTTPError(_FakeResponse(422, {"detail": [{"msg": "field required"}]}))
    )
    assert "field required" in validation
    assert "content filter" not in validation  # an ordinary 422 must NOT read as moderation
    # Nested error object, and a non-JSON body.
    nested = fal._fal_error_detail(_FakeResponse(400, {"error": {"message": "boom"}}))
    assert nested == "boom"
    assert fal._fal_error_detail(_FakeResponse(500, None, text="  gateway blew up  ")) == \
        "gateway blew up"
    assert fal._fal_error_detail(None) == ""
    # A non-HTTP failure keeps its own message.
    assert fal.fal_error_message(RuntimeError("The model returned no output.")) == \
        "The model returned no output."


def test_is_moderation_detail_discriminates() -> None:
    assert fal.is_moderation_detail("flagged by a content moderation policy")
    assert fal.is_moderation_detail("Request blocked by the safety system")
    assert not fal.is_moderation_detail("field required")
    assert not fal.is_moderation_detail("")


def test_fal_key_storage(tmp_path) -> None:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    assert store.fal_status() == {"configured": False, "encrypted": False}
    assert store.set_fal_key("  fal-secret  ")["configured"] is True
    assert store.fal_key() == "fal-secret"
    assert store.clear_fal_key()["configured"] is False
    assert store.fal_key() is None


def test_fal_status_uses_the_contract_field_name(tmp_path) -> None:
    """`ApiKeyStatus` (src/shared/types.ts) is the frozen wire contract: the renderer reads
    `status.configured`. Core previously returned `hasKey`, which resolved to `undefined`
    client-side - a saved key still showed "Not set" and the input just cleared. Pin the exact
    key set on every status-returning path so the two sides can't drift again."""
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    expected = {"configured", "encrypted"}
    assert set(store.fal_status()) == expected
    assert set(store.set_fal_key("fal-secret")) == expected
    assert set(store.clear_fal_key()) == expected


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
    # Untagged inputs stay out of byHandle so they fall back to positional kind resolution.
    assert resolved["byHandle"] == {}


def test_resolve_fal_inputs_groups_by_handle(tmp_path) -> None:
    """Two images on different ports (a start and an end keyframe) stay distinguishable."""
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    project = store.create_project("Fal")
    conn, folder = store.conn(), store.folder()
    (folder / "assets").mkdir(exist_ok=True)
    for name in ("start", "end"):
        (folder / "assets" / f"{name}.png").write_bytes(name.encode())
        conn.execute(
            "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
            f"VALUES ('{name}', ?, '{name}', 'assets/{name}.png', 'image', 0)",
            (project["id"],),
        )
    gen = mb.add_gen_node(conn, "minimax/h3/image-to-video", 0, 0, kind="video", params={},
                          title="H3")
    fr.add_input(conn, gen["frameId"], "start", "image")
    fr.add_input(conn, gen["frameId"], "end", "end_image")

    resolved = fal.resolve_fal_inputs(conn, folder, gen["frameId"])
    assert len(resolved["images"]) == 2
    assert resolved["byHandle"]["image"] == [resolved["images"][0]]
    assert resolved["byHandle"]["end_image"] == [resolved["images"][1]]


def test_the_same_asset_can_feed_two_ports(tmp_path) -> None:
    """Dedup is per (asset, handle): one image wired to both keyframes must not collapse to one."""
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    project = store.create_project("Fal")
    conn = store.conn()
    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('a1', ?, 'in', 'assets/in.png', 'image', 0)",
        (project["id"],),
    )
    gen = mb.add_gen_node(conn, "minimax/h3/image-to-video", 0, 0, kind="video", params={},
                          title="H3")
    first = fr.add_input(conn, gen["frameId"], "a1", "image")
    second = fr.add_input(conn, gen["frameId"], "a1", "end_image")
    assert first["id"] != second["id"]
    # Re-adding the same (asset, handle) pair still dedups.
    assert fr.add_input(conn, gen["frameId"], "a1", "image")["id"] == first["id"]


def test_fal_active_reports_runs_in_flight() -> None:
    """Same contract as the Core side, since the handler merges both into one queue."""

    class _Events:
        def broadcast(self, channel: str, payload: dict) -> None: ...

    gen = fal.FalGeneration(store=None, events=_Events())
    gen._active["frame-1"] = True
    gen._progress("frame-1", 0.25, "Queued")

    assert gen.active() == [{"frameId": "frame-1", "fraction": 0.25, "status": "Queued"}]

    gen.cancel("frame-1")
    assert gen.active() == []


def test_a_completed_request_can_still_carry_a_rejection_instead_of_output() -> None:
    """Seedance answers a reference image carrying a face with a 200 whose body is the refusal,
    while the queue reports COMPLETED. Reading only the queue told the user the model returned
    nothing, which points at a bug rather than at the reference they have to swap."""

    class _Response:
        def __init__(self, payload: object) -> None:
            self._payload = payload
            self.text = str(payload)

        def json(self) -> object:
            return self._payload

    refusal = _Response(
        {
            "detail": [
                {
                    "loc": ["body", "image_urls"],
                    "msg": "The images or videos provided may contain likenesses of real people "
                    "or other private information that cannot be processed.",
                    "type": "content_policy_violation",
                }
            ]
        }
    )
    message = fal._no_output_message(refusal)
    assert "content filter" in message
    assert "likenesses of real people" in message

    # An empty body still says something, rather than inventing a reason it does not have.
    assert fal._no_output_message(_Response({})) == "The model returned no output."

    # A non-moderation reason is passed through rather than dressed up as a filter block.
    other = _Response({"detail": "the upstream model timed out"})
    assert fal._no_output_message(other) == (
        "The model returned no output. The provider said: the upstream model timed out"
    )


def _character_board(source_type: str, params: dict) -> dict:
    return {
        "items": [
            {"id": "src", "type": "core",
             "data": {"core": {"type": source_type, "params": params}}},
            {"id": "gen", "type": "frame", "frameId": "F"},
        ],
        "connectors": [
            {"fromItemId": "src", "toItemId": "gen", "data": {"targetHandle": "character"}}
        ],
    }


def test_a_character_reaches_a_fal_node_from_write_as_well_as_load(monkeypatch) -> None:
    """Write .char is the node a user has in hand right after building a character, and it names the
    file it just saved under `filename` rather than `file`. Reading only Load's param meant the
    character silently never reached the request and the endpoint rendered a stranger."""
    monkeypatch.setattr(
        fal.mb, "list_board",
        lambda conn: _character_board("character/write", {"filename": "  /models/emmy-s500-v6"}),
    )
    # A typed path and a missing suffix both reduce to what `library.resolve` accepts.
    assert fal.wired_character(object(), "F") == "emmy-s500-v6.char"

    monkeypatch.setattr(
        fal.mb, "list_board",
        lambda conn: _character_board("character/load", {"file": "emmy-s500-v6.char"}),
    )
    assert fal.wired_character(object(), "F") == "emmy-s500-v6.char"


def test_an_unsaved_character_says_so_rather_than_rendering_a_stranger(monkeypatch) -> None:
    """Every other character node emits an identity that exists only inside a running graph, and a
    fal request cannot execute one. Silently dropping it is the worst outcome: the render succeeds
    and looks like the character simply did not work."""
    monkeypatch.setattr(
        fal.mb, "list_board",
        lambda conn: _character_board("character/encode", {"name": "emmy"}),
    )
    with pytest.raises(ValueError, match="Write .char first"):
        fal.wired_character(object(), "F")


def test_no_character_wire_is_not_an_error(monkeypatch) -> None:
    board = _character_board("character/load", {"file": "x.char"})
    board["connectors"] = []
    monkeypatch.setattr(fal.mb, "list_board", lambda conn: board)
    assert fal.wired_character(object(), "F") is None
