"""Characters on MiniMax H3: the reference policy, the prompt form, and scoring a video take."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from inline_core.characters import charfile as cf
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


def _scored(value: float, subject: bool = True, face: bool = True) -> dict[str, Any]:
    return {"score": value, "subjectCounted": subject, "faceBearing": face}


def _frames(count: int) -> list[tuple[float, str]]:
    """`_sample_frames` carries each frame's timestamp, so a clip's worst moment can be found."""
    return [(float(i), "f") for i in range(count)]


def test_a_video_score_is_the_median_of_the_frames_that_measured(monkeypatch) -> None:
    """A mean lets one blurred frame drag the number; a median survives one bad and one lucky."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: _frames(5))
    scores = iter([_scored(80), _scored(12), _scored(78), _scored(82), _scored(79)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["score"] == 79.0
    assert out["frames"] == 5
    # The median alone would hide the dip that the eye sees, so the worst frame rides with it.
    assert out["min"] == 12.0 and out["minAt"] == 1.0


def test_a_frame_with_no_face_drops_out_rather_than_scoring_zero(monkeypatch) -> None:
    """`score` returns None for unmeasurable, which is not the same claim as a score of nothing."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: _frames(3))
    scores = iter([_scored(90), None, _scored(70)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["frames"] == 2, "the unmeasurable frame is not counted"
    assert out["score"] == 80.0
    # Counted rather than silently dropped: how much of the clip could not be read is a fact.
    assert out["noFace"] == 1


def test_a_frame_scored_only_by_the_subject_term_is_not_identity(monkeypatch) -> None:
    """With no face, `score` falls back to DINOv2 alone - which measures framing and setting, and
    must never decide identity. Counting it would let a turned head lower the identity number."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: _frames(3))
    scores = iter([_scored(90), _scored(11, face=False), _scored(70)])
    monkeypatch.setattr(mod.scoring, "score", lambda *_a, **_k: next(scores))

    out = mod._score_video(Path("clip.mp4"), {}, [], [], [])
    assert out is not None and out["frames"] == 2 and out["noFace"] == 1
    assert out["min"] == 70.0, "the subject-only 11 never reaches the identity statistics"


def test_one_face_only_frame_makes_the_whole_score_face_only(monkeypatch) -> None:
    """Reporting a blended number when a frame's subject term was noise hides the dropped term."""
    from inline_core.studio import characters as mod

    monkeypatch.setattr(mod, "_sample_frames", lambda *_a, **_k: _frames(2))
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

    def fake(
        chosen: str, arch: str = "", prefer: str | None = None,
        limit: int | None = None, **_: Any,
    ):
        seen["arch"], seen["prefer"], seen["limit"] = arch, prefer, limit
        return None

    monkeypatch.setattr(characters, "char_apply", fake)
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    ref = next(v for v in VARIANTS if v.references)
    _apply_character(
        {"character": [type("I", (), {"file": "x.char"})()]}, ref, {}
    )
    # The cap travels with the call: `char_apply` divides the slots by role, which trimming the
    # returned list could not do without knowing what each reference is of.
    assert seen == {"arch": "minimax-h3", "prefer": "reference", "limit": 9}


def test_a_node_with_no_reference_channel_takes_whatever_the_character_prefers(monkeypatch) -> None:
    from inline_core.characters import apply as characters

    seen: dict[str, object] = {}

    def fake(
        chosen: str, arch: str = "", prefer: str | None = None,
        limit: int | None = None, **_: Any,
    ):
        seen["prefer"], seen["limit"] = prefer, limit
        return None

    monkeypatch.setattr(characters, "char_apply", fake)
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    fl2va = next(v for v in VARIANTS if not v.references)
    _apply_character(
        {"character": [type("I", (), {"file": "x.char"})()]}, fl2va, {}
    )
    assert seen["prefer"] is None
    # No reference channel, so no cap to state.
    assert seen["limit"] is None


def test_prefer_overrides_the_adapter_default() -> None:
    """`char_apply`'s own rule is adapter-wins; `prefer` is what a node uses to say it cannot."""
    import inspect

    from inline_core.characters.apply import char_apply

    assert "prefer" in inspect.signature(char_apply).parameters


def test_a_character_with_more_references_than_the_model_takes_is_trimmed() -> None:
    """H3 takes 9 images; a character built for another model may carry more. Refusing sent a user
    to unwire images they had not wired, because every one of them came from the character."""
    from inline_core.characters.apply import _fit_roles

    refs = [f"r{i}" for i in range(12)]
    roles = [cf.ROLE_FACE] * 12
    kept, kept_roles = _fit_roles(refs, roles, 9)
    assert len(kept) == len(kept_roles) == 9
    assert kept == refs[:9], "order is the prompt's numbering, so it has to be preserved"


def test_trimming_divides_the_slots_by_role() -> None:
    """Cutting the tail dropped whichever role happened to be last: a character with wardrobe lost
    its cloth references on every model that takes fewer than it holds."""
    from inline_core.characters.apply import _fit_roles

    roles = [cf.ROLE_FACE] * 6 + [cf.ROLE_BODY] * 4 + [cf.ROLE_CLOTH] * 3
    refs = [f"{r}{i}" for i, r in enumerate(roles)]
    _kept, kept_roles = _fit_roles(refs, roles, 9)
    counts = {role: kept_roles.count(role) for role in cf.ROLES}
    assert sum(counts.values()) == 9
    assert counts[cf.ROLE_CLOTH] > 0, "the last role must survive the cut"
    assert counts[cf.ROLE_FACE] >= counts[cf.ROLE_BODY] >= counts[cf.ROLE_CLOTH]


def test_the_prefix_never_names_a_reference_that_was_trimmed() -> None:
    """The prefix is what the prompt resolves; naming <Picture 10> when nine were sent addresses a
    position the model cannot see. Refs and roles are cut together, so it cannot drift."""
    from inline_core.characters.apply import AppliedCharacter, _fit_roles

    refs, roles = _fit_roles([f"r{i}" for i in range(12)], [cf.ROLE_FACE] * 12, 9)
    prefix = AppliedCharacter("Ada", refs, "freckles", roles=roles).prompt_prefix(1, style="token")
    assert "<Picture 9>" in prefix and "<Picture 10>" not in prefix


def test_wired_images_keep_priority_over_the_character(monkeypatch) -> None:
    """What the user wired is explicit; the character fills whatever room is left."""
    from inline_core.characters import apply as characters
    from inline_core.characters.apply import AppliedCharacter
    from inline_core.models.minimaxh3.runner import VARIANTS, _apply_character

    seen: dict[str, Any] = {}

    def fake(
        chosen: str, arch: str = "", prefer: str | None = None,
        limit: int | None = None, **_: Any,
    ):
        seen["limit"] = limit
        return AppliedCharacter("Ada", [f"r{i}" for i in range(limit or 0)], "freckles")

    monkeypatch.setattr(characters, "char_apply", fake)
    ref = next(v for v in VARIANTS if v.references)
    inputs = {
        "character": [type("I", (), {"file": "x.char"})()],
        "references": ["mine1", "mine2", "mine3"],
    }
    out = _apply_character(inputs, ref, {})
    assert seen["limit"] == 6, "3 wired leaves 6 of H3's 9 for the character"
    assert out is not None and len(out.refs) == 6
    assert out.prefix.startswith("<Picture 4>"), "and it is numbered after the wired ones"



def test_the_resolution_param_is_on_the_node_face_and_defaults_to_capping() -> None:
    """Default 1024, not uncapped, and named for what it does. It sets what the `.char` stores;
    H3 re-resizes every reference onto 2048 on the way in, so it buys disk and never VRAM."""
    from inline_core.models.character.runner import COMPILE_REFS

    field = next(p for p in COMPILE_REFS.params if p.key == "ref_resolution")
    assert field.label == "Stored Reference Resolution"
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



def test_an_encoder_oom_counts_what_the_model_sees_not_what_was_stored(monkeypatch) -> None:
    """The pipeline calls `resolve_reference_image_size` and puts every reference back onto a 2048
    short edge. Counting the stored pixels instead reported 1,280 tokens for a set that really cost
    20,480, and told the user to lower a setting that could not have helped."""
    import tempfile

    from PIL import Image

    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3.runner import Request, _oom, _reference_tokens
    from inline_core.models.references import ReferenceKind

    monkeypatch.setattr(rt, "foreign_vram_bytes", lambda *a, **k: 0)

    with tempfile.TemporaryDirectory() as tmp:
        counts = {}
        for stored in (512, 2048):
            paths = []
            for index in range(5):
                path = f"{tmp}/{stored}_{index}.png"
                Image.new("RGB", (stored, stored)).save(path)
                paths.append(path)
            refs = tuple(
                type("R", (), {"kind": ReferenceKind.IMAGE,
                               "value": type("V", (), {"path": p})()})()
                for p in paths
            )
            request = Request(
                prompt="", num_frames=124, width=544, height=768, num_inference_steps=20,
                seed=1, partition="ref2va", references=refs,
            )
            counts[stored] = _reference_tokens(request)[1]

        # The stored size is irrelevant: both are five references at an enforced 2048.
        assert counts[512] == counts[2048] == 5 * 64 * 64

        message = _oom(request)
        assert "20,480 vision tokens" in message
        assert "wire fewer references" in message
        # The two levers that cannot move this must not be offered as though they can.
        assert "960x544" not in message
        assert "Lower Resized Reference Resolution" not in message


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


def test_body_and_clothing_references_are_never_scored_against_the_face() -> None:
    """SFace measures faces. A body shot that happens to show one would be judged on the wrong
    thing and could be flagged as an outlier for it, so it is held out of scoring entirely."""
    from inline_core.characters import verify

    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    manifest.refs = [
        {"path": f"refs/{i:03d}.png", "sha256": f"h{i}", "role": role}
        for i, role in enumerate([cf.ROLE_FACE] * 3 + [cf.ROLE_BODY, cf.ROLE_CLOTH])
    ]
    doc = cf.CharDoc(manifest=manifest, members={})

    scored: list[int] = []

    def fake_images(_doc):
        return [object()] * len(manifest.refs)

    def fake_embed(image):
        scored.append(id(image))
        return [1.0, 0.0]

    import inline_core.characters.encode as enc
    import inline_core.characters.scoring as sc

    original_images, original_embed = enc.ref_images, sc.embed_face
    enc.ref_images, sc.embed_face = fake_images, fake_embed
    try:
        verdict = verify.verify(doc)
    finally:
        enc.ref_images, sc.embed_face = original_images, original_embed

    assert verdict.unscored == [3, 4], "the body and cloth refs, by position"
    assert len(scored) == 3, "only the three face refs reached the face encoder"
    assert 3 not in verdict.flagged and 4 not in verdict.flagged
    assert "not scored" in verdict.note


def test_the_verdict_says_body_references_are_unscored() -> None:
    """The UI has to be able to say it: a reference that is used but not measured is not the same
    as one that passed, and showing it as passing would be a claim nothing checked."""
    from inline_core.characters import verify

    verdict = verify.Verdict(mode=verify.MODE_BOOTSTRAP, floor=50.0, unscored=[2, 3])
    assert verdict.to_json()["unscored"] == [2, 3]


def test_removing_a_reference_keeps_the_unscored_positions_pointing_at_the_right_images() -> None:
    """Every list in a verdict is a position into `manifest.refs`, so a removal shifts them all.
    Remapping three of the four would leave `unscored` ringing whatever moved into its slot."""
    from inline_core.characters import verify

    verdict = verify.Verdict(
        mode=verify.MODE_BOOTSTRAP, floor=50.0,
        agreement=[90.0, 80.0, None, None],
        flagged=[1], unchecked=[], duplicates=[], unscored=[2, 3],
    )
    before = ["a.png", "b.png", "c.png", "d.png"]
    after = ["a.png", "c.png", "d.png"]  # "b.png" removed
    verify._reindex(verdict, before, after)
    assert verdict.unscored == [1, 2], "c and d kept their identity, one slot earlier"
    assert verdict.flagged == [], "the flagged reference is the one that went"


def test_a_role_line_refers_back_to_a_picture_without_re_declaring_it() -> None:
    """`<Picture N>` is H3's reserved label, emitted before each vision block. The role lines used
    to repeat it with nothing behind the second, and the model replayed the references as the
    opening frames of the video. Each label must be declared exactly once."""
    from inline_core.characters.apply import AppliedCharacter

    roles = [cf.ROLE_FACE] * 4 + [cf.ROLE_BODY] * 2 + [cf.ROLE_CLOTH] * 2
    refs = [f"r{i}" for i in range(len(roles))]
    character = AppliedCharacter("Ada", refs, "", roles=roles)
    prefix = character.prompt_prefix(1, style="token", role_lines=True)

    for n in range(1, len(roles) + 1):
        assert prefix.count(f"<Picture {n}>") == 1, f"<Picture {n}> is declared more than once"
    # The roles still have to be bound, just in prose that cannot be mistaken for a label.
    assert "Pictures 5 and 6 show Ada's full body and build." in prefix
    assert "Pictures 7 and 8 show Ada's outfit." in prefix


def test_the_role_line_switch_restores_the_prompt_a_character_had_before_roles() -> None:
    """Off, the bindings go and nothing else does, so the switch isolates one variable: the roles
    still decide which references are sent."""
    from inline_core.characters.apply import AppliedCharacter

    roles = [cf.ROLE_FACE] * 4 + [cf.ROLE_BODY] * 2 + [cf.ROLE_CLOTH] * 2
    refs = [f"r{i}" for i in range(len(roles))]
    with_roles = AppliedCharacter("Ada", refs, "freckles", roles=roles)
    # The same references with no roles recorded at all: a character written before the feature.
    without = AppliedCharacter("Ada", refs, "freckles")

    off = with_roles.prompt_prefix(1, style="token", role_lines=False)
    assert off == without.prompt_prefix(1, style="token")
    assert "full body and build" not in off
    assert "full body and build" in with_roles.prompt_prefix(1, style="token", role_lines=True)


def test_the_h3_node_leaves_the_role_lines_off_unless_asked() -> None:
    """A param that defaults on would ship the behaviour that replayed the references."""
    from inline_core.models.minimaxh3.runner import DESCRIPTORS, VARIANTS

    ref = next(v for v in VARIANTS if v.references)
    field = next(
        f for f in DESCRIPTORS[ref.node_type].params if f.key == "character_role_lines"
    )
    assert field.default is False


def test_the_node_defaults_to_every_reference_the_model_takes() -> None:
    """A smaller default would quietly change what every existing graph renders."""
    from inline_core.models.minimaxh3.runner import DESCRIPTORS, REFERENCE_LIMITS, VARIANTS

    ref = next(v for v in VARIANTS if v.references)
    params = {f.key: f for f in DESCRIPTORS[ref.node_type].params}
    assert params["character_references"].default == REFERENCE_LIMITS.max_images
    assert params["character_references"].max == REFERENCE_LIMITS.max_images
    assert params["character_reference_roles"].default == "all"


def test_a_full_reference_input_names_the_wiring_not_the_character(monkeypatch) -> None:
    """Nine wired images leave the character no slot, which is not the same fact as an empty .char.

    It used to raise "has no minimax-h3 references ... write it again", sending the user off to
    rebuild a character that was never the problem.
    """
    from inline_core.errors import ComponentError
    from inline_core.models.minimaxh3.runner import REFERENCE_LIMITS, VARIANTS, _apply_character

    ref = next(v for v in VARIANTS if v.references)
    inputs = {
        "character": [type("I", (), {"file": "x.char"})()],
        "references": ["img"] * REFERENCE_LIMITS.max_images,
    }
    with pytest.raises(ComponentError) as raised:
        _apply_character(inputs, ref, {})
    assert "no slot left" in str(raised.value)
    assert "Compile References" not in str(raised.value), "the character is not at fault"
