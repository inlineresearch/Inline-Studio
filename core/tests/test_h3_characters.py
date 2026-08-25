"""Characters on MiniMax H3: the reference policy, the prompt form, and scoring a video take."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from inline_core.characters import encode
from inline_core.characters.apply import AppliedCharacter

# --- the reference policy -------------------------------------------------------------------------


def _resize(size: tuple[int, int], arch: str) -> tuple[int, int]:
    pytest.importorskip("PIL")
    from PIL import Image

    out = encode.normalise_reference(Image.new("RGB", size), encode.reference_policy(arch))
    return out.width, out.height


def test_h3_is_offered_a_reference_payload_at_all() -> None:
    """One entry in this table was what forced every model but FLUX.2 down the adapter route."""
    assert encode.MINIMAX_H3_ARCH in encode.REFERENCE_POLICIES


def test_h3_scales_onto_a_short_edge_rather_than_under_an_area_cap() -> None:
    """FLUX.2 caps area and only ever shrinks; H3 scales the short side onto 2048 either way."""
    assert _resize((3000, 2000), encode.MINIMAX_H3_ARCH) == (3072, 2048)
    assert _resize((640, 640), encode.MINIMAX_H3_ARCH) == (2048, 2048)
    assert _resize((3000, 2000), encode.FLUX2_KLEIN_ARCH) == (1248, 832)


def test_h3_has_no_area_cap() -> None:
    """The vendored packer says a 4:1 reference is 8192x2048, so the payload must agree."""
    assert _resize((8000, 2000), encode.MINIMAX_H3_ARCH) == (8192, 2048)


def test_an_out_of_range_aspect_is_refused_while_compiling() -> None:
    """The vendored blocks raise on this mid-run; catching it here names the reference instead."""
    pytest.importorskip("PIL")
    from PIL import Image

    with pytest.raises(ValueError, match="within 1:4"):
        encode.normalise_reference(Image.new("RGB", (5000, 900)), encode.MINIMAX_H3_POLICY)


def test_both_policies_land_on_their_own_grid() -> None:
    for arch, grid in ((encode.FLUX2_KLEIN_ARCH, 16), (encode.MINIMAX_H3_ARCH, 32)):
        width, height = _resize((1234, 987), arch)
        assert width % grid == 0 and height % grid == 0


# --- the prompt form ------------------------------------------------------------------------------


def _character(refs: int) -> AppliedCharacter:
    return AppliedCharacter("Ada", ["ref"] * refs, "freckles, dark hair")


def test_h3_names_references_as_tokens_not_as_ordinal_prose() -> None:
    """H3 resolves `<Picture N>`; FLUX.2's "Images 1 and 2 show" names positions H3 cannot see."""
    prefix = _character(2).prompt_prefix(3, style="token")
    assert prefix.startswith("<Picture 3> <Picture 4> show Ada")
    assert "Images" not in prefix


def test_the_offset_is_where_the_character_lands_not_where_it_starts() -> None:
    """The character's images are appended after whatever the user wired, so the numbering has to
    continue rather than restart."""
    assert _character(1).prompt_prefix(5, style="token").startswith("<Picture 5> shows Ada")


def test_flux2_keeps_its_prose() -> None:
    assert _character(2).prompt_prefix(1).startswith("Images 1 and 2 show Ada")


def test_an_adapter_only_character_is_the_description_alone() -> None:
    """No references to name, so there is nothing positional to say in either style."""
    assert AppliedCharacter("Ada", [], "freckles").prompt_prefix(1, style="token") == "freckles "


# --- scoring a video take -------------------------------------------------------------------------


def _scored(value: float, subject: bool = True) -> dict[str, Any]:
    return {"score": value, "subjectCounted": subject}


def test_a_video_score_is_the_median_of_the_frames_that_measured(monkeypatch) -> None:
    """A mean lets one blurred frame drag the number; a median survives one bad and one lucky."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: ["f"] * 5)
    scores = iter([_scored(80), _scored(12), _scored(78), _scored(82), _scored(79)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["score"] == 79.0
    assert out["frames"] == 5


def test_a_frame_with_no_face_drops_out_rather_than_scoring_zero(monkeypatch) -> None:
    """`score` returns None for unmeasurable, which is not the same claim as a score of nothing."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: ["f"] * 3)
    scores = iter([_scored(90), None, _scored(70)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["frames"] == 2, "the unmeasurable frame is not counted"
    assert out["score"] == 80.0


def test_one_face_only_frame_makes_the_whole_score_face_only(monkeypatch) -> None:
    """Reporting a blended number when a frame's subject term was noise hides the dropped term."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: ["f"] * 2)
    scores = iter([_scored(80), _scored(70, subject=False)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["subjectCounted"] is False


def test_a_clip_that_cannot_be_read_scores_nothing(monkeypatch) -> None:
    """A missing ffmpeg means no score, never a failed render."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: [])
    assert mod._score_video(Path("clip.mp4"), {}, [], [], []) is None


# --- which route a node gets ----------------------------------------------------------------------


def test_the_reference_node_asks_for_references_over_an_adapter(tmp_path, monkeypatch) -> None:
    """A character carrying both defaults to its adapter, which leaves H3's reference partition with
    nothing to condition on: it refused the run saying no reference was wired, while one was."""
    from inline_core.characters import apply as characters

    seen: dict[str, object] = {}

    def fake(chosen: str, arch: str = "", prefer: str | None = None):
        seen["arch"], seen["prefer"] = arch, prefer
        return None

    monkeypatch.setattr(characters, "char_apply", fake)
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    ref = next(v for v in VARIANTS if v.references)
    _apply_character({"character": [type("I", (), {"file": "x.char"})()]}, ref)
    assert seen == {"arch": "minimax-h3", "prefer": "reference"}


def test_a_node_with_no_reference_channel_takes_whatever_the_character_prefers(monkeypatch) -> None:
    from inline_core.characters import apply as characters

    seen: dict[str, object] = {}

    def fake(chosen: str, arch: str = "", prefer: str | None = None):
        seen["prefer"] = prefer
        return None

    monkeypatch.setattr(characters, "char_apply", fake)
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    fl2va = next(v for v in VARIANTS if not v.references)
    _apply_character({"character": [type("I", (), {"file": "x.char"})()]}, fl2va)
    assert seen["prefer"] is None


def test_prefer_overrides_the_adapter_default() -> None:
    """`char_apply`'s own rule is adapter-wins; `prefer` is what a node uses to say it cannot."""
    import inspect

    from inline_core.characters.apply import char_apply

    assert "prefer" in inspect.signature(char_apply).parameters


def test_a_character_with_more_references_than_the_model_takes_is_trimmed(monkeypatch) -> None:
    """H3 takes 9 images; a character built for another model may carry more. Refusing sent a user
    to unwire images they had not wired, because every one of them came from the character."""
    from inline_core.characters import apply as characters
    from inline_core.characters.apply import AppliedCharacter
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    monkeypatch.setattr(
        characters, "char_apply",
        lambda *_a, **_k: AppliedCharacter("Ada", [f"r{i}" for i in range(10)], "freckles"),
    )
    ref = next(v for v in VARIANTS if v.references)
    out = _apply_character({"character": [type("I", (), {"file": "x.char"})()]}, ref)
    assert out is not None and len(out.refs) == 9


def test_the_prefix_never_names_a_reference_that_was_trimmed(monkeypatch) -> None:
    """The prefix is what the prompt resolves; naming <Picture 10> when nine were sent addresses a
    position the model cannot see."""
    from inline_core.characters import apply as characters
    from inline_core.characters.apply import AppliedCharacter
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    monkeypatch.setattr(
        characters, "char_apply",
        lambda *_a, **_k: AppliedCharacter("Ada", [f"r{i}" for i in range(10)], "freckles"),
    )
    ref = next(v for v in VARIANTS if v.references)
    out = _apply_character({"character": [type("I", (), {"file": "x.char"})()]}, ref)
    assert out is not None
    assert "<Picture 9>" in out.prefix and "<Picture 10>" not in out.prefix


def test_wired_images_keep_priority_over_the_character(monkeypatch) -> None:
    """What the user wired is explicit; the character fills whatever room is left."""
    from inline_core.characters import apply as characters
    from inline_core.characters.apply import AppliedCharacter
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    monkeypatch.setattr(
        characters, "char_apply",
        lambda *_a, **_k: AppliedCharacter("Ada", [f"r{i}" for i in range(10)], "freckles"),
    )
    ref = next(v for v in VARIANTS if v.references)
    inputs = {
        "character": [type("I", (), {"file": "x.char"})()],
        "references": ["mine1", "mine2", "mine3"],
    }
    out = _apply_character(inputs, ref)
    assert out is not None and len(out.refs) == 6, "3 wired + 6 from the character is the 9 cap"
    assert out.prefix.startswith("<Picture 4>"), "and it is numbered after the wired ones"



def test_the_resolution_param_is_on_the_node_face_and_defaults_to_capping() -> None:
    """Default 1024, not uncapped: H3's own policy is 2048, and a character compiled there is what
    put 36,864 vision tokens on the card."""
    from inline_core.models.character.runner import COMPILE_REFS

    field = next(p for p in COMPILE_REFS.params if p.key == "ref_resolution")
    assert field.label == "Resized Reference Resolution"
    assert field.default == 1024
    assert field.on_face is True
    assert field.min == encode.NO_REFERENCE_CAP


def test_the_cap_only_ever_lowers_a_model_policy() -> None:
    """It is a ceiling, not a target: raising H3 past 2048 or FLUX.2 past its area cap would ask
    each model for a size it does not accept."""
    h3, flux = encode.MINIMAX_H3_ARCH, encode.FLUX2_KLEIN_ARCH
    assert encode.capped_policy(h3, 1024)["short_edge"] == 1024
    assert encode.capped_policy(h3, 4096)["short_edge"] == 2048
    assert encode.capped_policy(flux, 2048)["max_pixels"] == 1024 * 1024
    assert encode.capped_policy(flux, 512)["max_pixels"] == 512 * 512
    # -1 means the model's own policy, which is the only safe reading of "no resize": H3's packer
    # requires the 32px grid, so a raw source size is not something it can be handed.
    for arch in (h3, flux):
        assert encode.capped_policy(arch, -1) == encode.reference_policy(arch)
        assert encode.capped_policy(arch, None) == encode.reference_policy(arch)


def test_a_graph_saved_before_the_param_existed_still_gets_the_cap() -> None:
    """Reading a missing param as uncapped would silently compile old graphs at 2048."""
    from inline_core.models.character.runner import _resolution

    assert _resolution(None) == 1024
    assert _resolution("") == 1024
    assert _resolution(2048) == 2048
    assert _resolution(-1) == encode.NO_REFERENCE_CAP
    assert _resolution(0) == encode.NO_REFERENCE_CAP


def test_capping_h3_to_1024_quarters_the_vision_tokens() -> None:
    """The arithmetic the whole param exists for."""
    from PIL import Image

    from inline_core.characters import charfile as cf

    manifest = cf.Manifest(char_id="c", name="c", created_at=0, modified_at=0)
    members: dict[str, bytes] = {}
    images = [Image.new("RGB", (3840, 2160)) for _ in range(2)]
    for index, image in enumerate(images):
        path = f"refs/ref_{index:03d}.png"
        members[path] = encode._png_bytes(image)
        manifest.refs.append({"path": path, "sha256": cf.sha256_bytes(members[path])})

    def tokens(cap: int) -> int:
        arch = encode.MINIMAX_H3_ARCH
        encode.build_payload(manifest, members, images, arch, encode.capped_policy(arch, cap))
        total = 0
        for entry in manifest.payloads[arch]["files"]:
            width, height = Image.open(io.BytesIO(members[entry["path"]])).size
            total += (width // 32) * (height // 32)
        return total

    assert tokens(1024) * 4 == tokens(2048)
    # A 4K source is not special: the policy resizes onto its target either way.
    assert tokens(-1) == tokens(2048)



def test_an_encoder_oom_points_at_the_character_not_the_canvas(monkeypatch) -> None:
    """The canvas hint sent a user to resize twice for nothing: references are encoded before any
    frame exists, so a 1344x768 -> 544x768 drop left the failing allocation byte-identical. The
    size that matters was fixed when the character was compiled, so that is what the error names.
    """
    import tempfile

    from PIL import Image

    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3.runner import Request, _oom
    from inline_core.models.references import ReferenceKind

    # Stubbed because it reads the live card otherwise, so this asserted on whatever else happened
    # to be running: it passed on an idle box and failed beside a training run.
    monkeypatch.setattr(rt, "foreign_vram_bytes", lambda *a, **k: 0)

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for index in range(9):
            path = f"{tmp}/ref{index}.png"
            Image.new("RGB", (2048, 2048)).save(path)
            paths.append(path)
        refs = tuple(
            type("R", (), {"kind": ReferenceKind.IMAGE, "value": type("V", (), {"path": p})()})()
            for p in paths
        )
        request = Request(
            prompt="", num_frames=144, width=544, height=768, num_inference_steps=50,
            seed=1, partition="ref2va", references=refs,
        )
        message = _oom(request)

    # Measured off the pixels, not off a setting this node no longer carries.
    assert "36,864 vision tokens" in message
    assert "Resized Reference Resolution" in message
    assert "does not affect this step" in message
    assert "960x544" not in message

    # With no references the canvas really is the lever, so that hint has to survive untouched.
    plain = Request(
        prompt="", num_frames=144, width=1344, height=768,
        num_inference_steps=50, seed=1, partition="fl2va",
    )
    assert "960x544" in _oom(plain)


def test_a_card_held_by_another_process_is_named_before_anything_on_this_node(monkeypatch) -> None:
    """A run with 5 references and 5,120 vision tokens was told to lower its reference resolution,
    while a training run held 29 of the card's 46 GB. Nothing on this node frees that, and every
    other hint sends the user to change a setting that was never the cause."""
    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3.runner import Request, _oom
    from inline_core.models.references import ReferenceKind

    refs = tuple(type("R", (), {"kind": ReferenceKind.IMAGE})() for _ in range(5))
    request = Request(
        prompt="", num_frames=144, width=544, height=768, num_inference_steps=50,
        seed=1, partition="ref2va", references=refs,
    )

    monkeypatch.setattr(rt, "foreign_vram_bytes", lambda *a, **k: 29 * 1024**3)
    busy = _oom(request)
    assert "29.0 GB" in busy and "another process" in busy
    assert "Resized Reference Resolution" not in busy, "do not blame the character"
    assert "960x544" not in busy, "do not blame the canvas"

    # Below the floor it is noise, and the reference hint is the useful one again.
    monkeypatch.setattr(rt, "foreign_vram_bytes", lambda *a, **k: 200 * 1024**2)
    assert "Resized Reference Resolution" in _oom(request)

    # A host-RAM exhaustion is never explained by another process's VRAM.
    monkeypatch.setattr(rt, "foreign_vram_bytes", lambda *a, **k: 29 * 1024**3)
    assert "another process" not in _oom(request, host=True)


def test_a_failed_run_releases_the_card_instead_of_poisoning_the_next(monkeypatch) -> None:
    """`free_vram` drops unused blocks and leaves the pipeline resident, so one OOM left ~43 GB
    held and every retry started from a full card - the same error forever, whatever was changed.
    A VRAM failure has to evict, and say so, rather than blame the character again."""
    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3.runner import Request, _oom
    from inline_core.models.references import ReferenceKind

    monkeypatch.setattr(rt, "foreign_vram_bytes", lambda *a, **k: 0)
    refs = tuple(type("R", (), {"kind": ReferenceKind.IMAGE})() for _ in range(5))
    request = Request(
        prompt="", num_frames=141, width=544, height=768, num_inference_steps=50,
        seed=1, partition="ref2va", references=refs,
    )

    held = _oom(request, held=int(43.5 * 1024**3))
    assert "43.5 GB already held by this render" in held
    assert "released" in held and "run it again" in held
    assert "Resized Reference Resolution" not in held, "do not blame the character"

    # An empty card is the case where the reference hint is the useful one.
    assert "Resized Reference Resolution" in _oom(request, held=0)
    # And host exhaustion is never explained by VRAM the render was holding.
    assert "already held" not in _oom(request, host=True, held=int(43.5 * 1024**3))


def test_the_runner_clears_the_pipeline_cache_on_a_vram_failure() -> None:
    """Emptying the allocator is not enough: the cached pipeline pins the weights themselves."""
    import inspect

    from inline_core.models.minimaxh3 import runner

    source = inspect.getsource(runner.MiniMaxH3Runner.run)
    handler = source[source.index("except torch.cuda.OutOfMemoryError") :]
    handler = handler.split("except MemoryError")[0]
    assert "PIPELINES.clear()" in handler
    # And the frame has to stop naming the pipeline first: `raise ... from error` keeps the
    # traceback, which keeps these locals, so evicting the cache alone frees nothing. Measured:
    # the card still held 43.5 GB after a failed run that did call clear().
    assert "pipe = None" in handler
    assert handler.index("pipe = None") < handler.index("PIPELINES.clear()")
