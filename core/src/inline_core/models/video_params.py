"""What a video model can be asked for: its frame grid, its canvas rule, and the params that follow.

Torch-free and pure, the way ``models/sampling.py``'s data layer is, so a runner splices its params
in with one call and the awkward arithmetic is unit-testable with no GPU and no weights.

A video model does not accept an arbitrary duration. Its VAE decodes in blocks, so only certain
frame counts exist, and a request snaps **up** onto the nearest one and is then clamped into the
model's duration window. Snapping up alone is not safe - the next count up can exceed the model's
own maximum - and snapping down alone silently shortens every request, so it takes both.

Training rounds the other way, and deliberately: see ``training/arch.py``'s ``ClipGrid``. A request
for a duration should be honoured where it is legal; a clip cannot be asked for frames the file
never held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import ComponentError
from ..graph.descriptor import ParamField, Widget


@dataclass(frozen=True)
class VideoGrid:
    """The frame counts a model can actually decode, as ``grid * n + offset`` within a window.

    MiniMax H3 is ``grid=17, offset=5`` at 24 fps between 5 and 15 seconds, so 124 to 345 frames.
    """

    fps: float
    grid: int
    offset: int
    min_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        if self.grid < 1 or self.fps <= 0:
            raise ValueError("A video grid needs a positive fps and a grid of at least 1.")
        if self.max_frames < self.min_frames:
            raise ValueError("This grid has no valid frame count inside its duration window.")

    @property
    def min_frames(self) -> int:
        """The smallest grid point at or above ``min_seconds``."""
        wanted = self.min_seconds * self.fps
        return self.grid * math.ceil((wanted - self.offset) / self.grid) + self.offset

    @property
    def max_frames(self) -> int:
        """The largest grid point at or below ``max_seconds``."""
        wanted = self.max_seconds * self.fps
        return self.grid * math.floor((wanted - self.offset) / self.grid) + self.offset

    def seconds(self, frames: int) -> float:
        return frames / self.fps


def snap_frames(seconds: float, grid: VideoGrid) -> int:
    """The frame count a request for ``seconds`` actually renders.

    Snaps **up** to the nearest grid point, matching both reference implementations, then clamps
    into the duration window. Rounding up alone is not safe: H3 at 14.9 seconds rounds to 362
    frames, which is 15.083 seconds and past the model's own 15 second limit. Clamping afterwards
    keeps the request honoured where it can be and legal where it cannot, so 10 seconds gives
    10.125 rather than 9.417, and 14.9 gives 14.375 rather than an error.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ComponentError("Video duration must be a positive number of seconds.")
    wanted = seconds * grid.fps
    n = math.ceil((wanted - grid.offset) / grid.grid)
    frames = grid.grid * n + grid.offset
    return max(grid.min_frames, min(grid.max_frames, frames))


def snap_canvas(
    width: int, height: int, *, multiple: int = 32, minimum: int = 0
) -> tuple[int, int]:
    """Round a canvas to the multiple a model's VAE needs, never below ``minimum``."""
    if multiple < 1:
        raise ValueError("A canvas multiple must be at least 1.")

    def one(value: int) -> int:
        rounded = int(round(value / multiple)) * multiple
        return max(multiple, max(minimum, rounded))

    return one(width), one(height)


def canvas_for_aspect(
    aspect: float, *, short_edge: int, multiple: int = 32, max_long_edge: int | None = None
) -> tuple[int, int]:
    """A model's native canvas for an aspect ratio (width / height), snapped to its multiple.

    Used when no explicit size is given, so the canvas follows a wired keyframe rather than a
    default that would letterbox it.
    """
    if not math.isfinite(aspect) or aspect <= 0:
        aspect = 16 / 9
    if aspect >= 1:
        height, width = short_edge, int(round(short_edge * aspect))
    else:
        width, height = short_edge, int(round(short_edge / aspect))
    if max_long_edge is not None:
        width, height = min(width, max_long_edge), min(height, max_long_edge)
    return snap_canvas(width, height, multiple=multiple, minimum=multiple)


def video_param_fields(
    grid: VideoGrid,
    *,
    short_edge: int,
    multiple: int = 32,
    max_long_edge: int | None = None,
    default_aspect: float = 16 / 9,
    canvas_hint: str = "",
) -> tuple[ParamField, ...]:
    """Duration and canvas params for a video node.

    Duration is exposed in **seconds**, not frames: a grid of ``17n + 5`` is an implementation
    detail of the VAE and nobody composes a shot in it. ``fps`` is deliberately absent - it is a
    model constant, and letting it be edited only desyncs it from the frame grid above.
    """
    width, height = canvas_for_aspect(
        default_aspect, short_edge=short_edge, multiple=multiple, max_long_edge=max_long_edge
    )
    size_label = f" ({canvas_hint})" if canvas_hint else ""
    return (
        ParamField(
            "duration",
            f"Duration (seconds, {grid.min_seconds:g} to {grid.max_seconds:g})",
            Widget.NUMBER,
            round(grid.seconds(grid.min_frames), 2),
            min=grid.min_seconds,
            max=grid.max_seconds,
            step=0.5,
        ),
        ParamField(
            "width", f"Width{size_label}", Widget.NUMBER, width,
            min=multiple, max=max_long_edge or 4096, step=multiple,
        ),
        ParamField(
            "height", f"Height{size_label}", Widget.NUMBER, height,
            min=multiple, max=max_long_edge or 4096, step=multiple,
        ),
    )
