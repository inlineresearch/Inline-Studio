"""`ClipGrid` replaced a call into MiniMax H3's vendored packing code. This proves it moved nothing.

The generalisation exists so LTX-2.5 can declare its own `8n + 1` grid instead of a second hardcoded
branch. That is only safe if the generic snap is the same function H3 was already getting, for every
input a clip can have - so this compares them directly rather than restating the formula.
"""

from __future__ import annotations

import pytest

from inline_core.training import arch as archs


def test_the_generic_snap_matches_h3s_vendored_one() -> None:
    from inline_core.models.minimaxh3.vendor.packing_ref2va import trim_reference_num_frames

    grid = archs.ARCHS[archs.MINIMAX_H3].clip
    assert grid is not None
    for frames in range(1, 601):
        assert grid.snap(frames) == trim_reference_num_frames(frames), frames


def test_the_h3_grid_still_reads_as_24fps_17n_plus_5() -> None:
    from inline_core.models.minimaxh3.vendor.packing import (
        MINIMAX_H3_FPS,
        MINIMAX_H3_FRAMES_PER_CHUNK,
        MINIMAX_H3_LATENTS_PER_CHUNK,
    )

    grid = archs.ARCHS[archs.MINIMAX_H3].clip
    assert grid is not None
    assert (grid.fps, grid.grid, grid.offset) == (
        MINIMAX_H3_FPS,
        MINIMAX_H3_FRAMES_PER_CHUNK,
        MINIMAX_H3_LATENTS_PER_CHUNK,
    )
    assert grid.min_frames == 22


def test_a_still_arch_reports_one_frame() -> None:
    for key in (archs.Z_IMAGE, archs.KREA2, archs.FLUX2):
        assert archs.ARCHS[key].clip is None
        assert archs.clip_frames(archs.ARCHS[key], 5.0) == 1


def test_a_clip_shorter_than_one_chunk_rounds_up_to_the_floor() -> None:
    """Refusing would be worse: the VAE simply cannot encode less than a chunk plus the head."""
    grid = archs.ARCHS[archs.MINIMAX_H3].clip
    assert grid is not None
    assert grid.snap(1) == grid.min_frames
    assert grid.snap(21) == grid.min_frames


def test_zero_frames_is_a_programming_error_not_a_clamp() -> None:
    grid = archs.ARCHS[archs.MINIMAX_H3].clip
    assert grid is not None
    with pytest.raises(ValueError):
        grid.snap(0)


def test_snapping_never_rounds_up_past_what_the_clip_holds() -> None:
    """The direction is the whole point: generation rounds up, training must not."""
    for key, arch in archs.ARCHS.items():
        if arch.clip is None:
            continue
        for frames in range(arch.clip.min_frames, 400):
            assert arch.clip.snap(frames) <= frames, (key, frames)
