"""The three LTX-2.5 node descriptors, and the arithmetic behind a request.

Everything here runs with no weights and no GPU. What it is really guarding is the two numbers a
unit test can check and a render cannot explain: the frame grid, and a canvas that has to survive
being halved for stage 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inline_core.errors import ComponentError
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind
from inline_core.models.ltx25 import runner as ltx


def variant(node_type: str) -> ltx.Variant:
    return next(v for v in ltx.VARIANTS if v.node_type == node_type)


def params(**overrides: Any) -> dict[str, Any]:
    return {**ltx.DESCRIPTORS[ltx.VARIANTS[0].node_type].defaults(), **overrides}


def inputs(prompt: str = "a cat", **wired: Any) -> dict[str, list[Any]]:
    return {"prompt": [prompt], **{k: [v] for k, v in wired.items()}}


# --- descriptors ---------------------------------------------------------------------------------


def test_three_nodes_all_produce_video_and_audio() -> None:
    assert len(ltx.VARIANTS) == 3
    for v in ltx.VARIANTS:
        d = ltx.DESCRIPTORS[v.node_type]
        assert d.output_kind is MediaKind.VIDEO
        assert [p.id for p in d.outputs] == ["video", "audio"]
        assert d.category == "Generate"


def test_every_node_takes_a_lora() -> None:
    """The point of shipping the dev transformer is that a trained LoRA loads on all three."""
    for v in ltx.VARIANTS:
        kinds = {p.id: p.kind for p in ltx.DESCRIPTORS[v.node_type].inputs}
        assert kinds["lora"] is PortKind.LORA


def test_keyframe_ports_follow_the_variant() -> None:
    ports = {v.node_type: {p.id for p in ltx.DESCRIPTORS[v.node_type].inputs} for v in ltx.VARIANTS}
    assert "image" not in ports["lightricks/ltx-2-5-text-to-video"]
    assert "image" in ports["lightricks/ltx-2-5-image-to-video"]
    assert "last_image" not in ports["lightricks/ltx-2-5-image-to-video"]
    assert "last_image" in ports["lightricks/ltx-2-5-first-last-frame"]


def test_fps_is_not_a_param() -> None:
    """It is a model constant. Editing it would only desync it from the frame grid."""
    keys = {p.key for p in ltx.DESCRIPTORS[ltx.VARIANTS[0].node_type].params}
    assert "fps" not in keys and "frame_rate" not in keys


# --- the frame grid ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "frames"),
    [
        (5.0, 121),  # upstream's own default clip length
        (1.0, 25),  # the floor
        (10.0, 241),
        (0.1, 25),  # below the window, clamped up rather than refused
        (30.0, 473),  # above the window, clamped down rather than raising
    ],
)
def test_duration_snaps_onto_the_eight_plus_one_grid(seconds: float, frames: int) -> None:
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"), params(duration=seconds), inputs()
    )
    assert request.num_frames == frames
    assert (request.num_frames - 1) % 8 == 0


def test_the_canvas_survives_being_halved() -> None:
    """Stage 1 renders at half size and LTX requires a multiple of 32 at *both* stages, so the size
    the node asks for has to be a multiple of 64."""
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"),
        params(width=1000, height=500),
        inputs(),
    )
    assert request.width % 64 == 0 and request.height % 64 == 0
    stage_1_w, stage_1_h = request.stage_1_size
    assert stage_1_w % 32 == 0 and stage_1_h % 32 == 0


def test_the_default_canvas_is_legal() -> None:
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"), params(), inputs()
    )
    assert request.width % 64 == 0 and request.height % 64 == 0


# --- conditioning --------------------------------------------------------------------------------


def test_text_to_video_conditions_on_nothing() -> None:
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"), params(), inputs()
    )
    assert request.conditionings == ()


def test_the_last_frame_pins_the_final_index(tmp_path: Path) -> None:
    first, last = tmp_path / "a.png", tmp_path / "b.png"
    first.write_bytes(b""), last.write_bytes(b"")
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-first-last-frame"),
        params(duration=5.0),
        inputs(image=_asset(first), last_image=_asset(last)),
    )
    assert [c.frame_index for c in request.conditionings] == [0, request.num_frames - 1]
    assert request.num_frames == 121


def test_image_to_video_without_an_image_is_a_wiring_error() -> None:
    with pytest.raises(ComponentError, match="First frame"):
        ltx.build_request(
            variant("lightricks/ltx-2-5-image-to-video"), params(), inputs()
        )


def test_a_prompt_is_required() -> None:
    with pytest.raises(ComponentError, match="prompt"):
        ltx.build_request(
            variant("lightricks/ltx-2-5-text-to-video"), params(), {"prompt": []}
        )


def test_call_kwargs_pass_an_already_snapped_frame_count(tmp_path: Path) -> None:
    """Upstream snaps **down** onto the same grid our params snap **up** onto. Passing a count that
    is already on the grid makes their snap a no-op; passing a raw one would quietly render fewer
    frames than the duration field promised."""
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"), params(duration=5.0), inputs()
    )
    call = ltx.call_kwargs(request, tmp_path)
    assert call["num_frames"] == 121
    assert (call["num_frames"] - 1) % 8 == 0
    assert call["frame_rate"] == ltx.GRID.fps


# --- modes ---------------------------------------------------------------------------------------


def test_fast_mode_ignores_the_step_count() -> None:
    """The distilled build runs a fixed 8 + 4 sigma schedule, so a step control there is a lie."""
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"),
        params(mode=ltx.MODE_FAST, num_inference_steps=77),
        inputs(),
    )
    assert request.num_inference_steps == ltx.DISTILLED_STEPS
    assert request.build == "distilled"


def test_quality_mode_honours_the_step_count_and_loads_dev() -> None:
    request = ltx.build_request(
        variant("lightricks/ltx-2-5-text-to-video"),
        params(mode=ltx.MODE_QUALITY, num_inference_steps=40),
        inputs(),
    )
    assert request.num_inference_steps == 40
    assert request.build == "dev"


class _Asset:
    def __init__(self, path: Path) -> None:
        self.path = str(path)


def _asset(path: Path) -> Any:
    return _Asset(path)


def test_a_yielded_chunk_is_split_into_frames() -> None:
    """The pipeline yields batches, not frames: a 2 second clip arrives as one (49,576,960,3)
    tensor. Wrapping a batch as a single frame reaches the encoder before it fails."""
    torch = pytest.importorskip("torch")
    from inline_core.models.ltx25.pipeline import _frames_in

    batch = torch.zeros(49, 8, 8, 3)
    assert len(_frames_in(batch)) == 49
    assert _frames_in(batch)[0].shape == (8, 8, 3)

    single = torch.zeros(8, 8, 3)
    assert len(_frames_in(single)) == 1
    assert len(_frames_in([single, single])) == 2
