"""The frame grid and canvas rules every video node snaps a request onto.

The H3 numbers here are the ones the model's own documentation calls out, including the case that
motivated clamping after the round-up: both reference implementations round up, and one of them
overshoots the model's own duration limit doing it.
"""

from __future__ import annotations

import pytest

from inline_core.errors import ComponentError
from inline_core.models.video_params import (
    VideoGrid,
    canvas_for_aspect,
    snap_canvas,
    snap_frames,
    video_param_fields,
)

#: MiniMax H3: 24 fps, blocks of 17 frames plus 5, between 5 and 15 seconds.
H3 = VideoGrid(fps=24.0, grid=17, offset=5, min_seconds=5.0, max_seconds=15.0)
#: A different shape entirely, so nothing below is fitted to one model.
WAN = VideoGrid(fps=16.0, grid=4, offset=1, min_seconds=1.0, max_seconds=5.0)


def test_h3_window_is_124_to_345_frames() -> None:
    assert H3.min_frames == 124  # 17*7+5, the first grid point at or above 5s
    assert H3.max_frames == 345  # 17*20+5, the last at or below 15s
    assert H3.seconds(345) == pytest.approx(14.375)


def test_every_grid_point_in_the_window_is_valid() -> None:
    for grid in (H3, WAN):
        for frames in range(grid.min_frames, grid.max_frames + 1):
            if (frames - grid.offset) % grid.grid:
                continue
            assert grid.min_seconds <= grid.seconds(frames) <= grid.max_seconds


def test_snapping_matches_the_reference_implementations() -> None:
    """Both ComfyUI's frame expression and the diffusers docs round up, so a 10 second request
    renders 10.125 rather than 9.417."""
    assert snap_frames(10.0, H3) == 243
    assert H3.seconds(243) == pytest.approx(10.125)


def test_a_request_near_the_maximum_is_clamped_not_overshot() -> None:
    """Rounding up alone gives 362 frames, i.e. 15.083s, past the model's own 15 second limit.
    ComfyUI's expression does exactly that; clamping afterwards keeps it legal."""
    assert snap_frames(14.9, H3) == 345
    assert H3.seconds(snap_frames(14.9, H3)) < H3.max_seconds


def test_requests_clamp_into_the_window() -> None:
    assert snap_frames(0.5, H3) == H3.min_frames  # below the floor
    assert snap_frames(60.0, H3) == H3.max_frames  # far above the ceiling
    assert snap_frames(5.0, H3) == 124  # exactly the floor still lands on a real grid point


def test_snapping_lands_on_a_grid_point_at_or_above_the_request() -> None:
    for seconds in (6.0, 7.3, 9.9, 12.5, 14.0):
        frames = snap_frames(seconds, H3)
        assert H3.seconds(frames) >= seconds
        assert (frames - H3.offset) % H3.grid == 0


def test_a_second_grid_behaves_the_same_way() -> None:
    assert WAN.min_frames == 17 and WAN.max_frames == 77
    assert snap_frames(3.0, WAN) == 49  # 4*12+1, i.e. 3.0625s
    assert (snap_frames(3.0, WAN) - WAN.offset) % WAN.grid == 0


def test_a_nonsense_duration_is_rejected() -> None:
    with pytest.raises(ComponentError):
        snap_frames(0, H3)
    with pytest.raises(ComponentError):
        snap_frames(float("nan"), H3)


def test_a_grid_with_no_valid_point_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="no valid frame count"):
        VideoGrid(fps=24.0, grid=100, offset=0, min_seconds=1.0, max_seconds=1.1)


# --- canvas -------------------------------------------------------------------------------------


def test_canvas_rounds_to_the_multiple_and_never_collapses() -> None:
    assert snap_canvas(1000, 500, multiple=32) == (992, 512)
    assert snap_canvas(1, 1, multiple=32) == (32, 32)


def test_canvas_for_aspect_puts_the_short_edge_on_the_short_side() -> None:
    wide = canvas_for_aspect(16 / 9, short_edge=768, multiple=32, max_long_edge=1344)
    tall = canvas_for_aspect(9 / 16, short_edge=768, multiple=32, max_long_edge=1344)
    assert wide == (1344, 768) and tall == (768, 1344)
    assert canvas_for_aspect(1.0, short_edge=768, multiple=32) == (768, 768)


def test_canvas_for_aspect_survives_a_degenerate_ratio() -> None:
    assert canvas_for_aspect(0.0, short_edge=768, multiple=32, max_long_edge=1344) == (1344, 768)


# --- the param fields -----------------------------------------------------------------------------


def test_param_fields_expose_seconds_and_a_canvas_but_never_fps() -> None:
    fields = {f.key: f for f in video_param_fields(H3, short_edge=768, max_long_edge=1344)}
    assert set(fields) == {"duration", "width", "height"}
    assert "fps" not in fields  # a model constant; editing it only desyncs the frame grid
    assert fields["duration"].min == 5.0 and fields["duration"].max == 15.0
    assert (fields["width"].default, fields["height"].default) == (1344, 768)
    assert fields["width"].step == 32
