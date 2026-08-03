"""Ordered multimodal references: the wired image, video and audio ports as one numbered list.

Order is meaning, not decoration. A reference-conditioned model addresses its inputs by position, so
the number on the node's face has to be the number the prompt resolves. That ordering runs
from ``moodboard.list_board`` (connectors by ``created_at``) through ``graph_build._edges_for``
into the ``inputs`` a runner receives. This module is the last link: it preserves the per-port
order it is handed and lays the ports out in a declared, stable sequence.

Torch-free: decoding belongs to ``models/video_runtime.py``. This only orders, numbers and counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..errors import ComponentError


class ReferenceKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


#: How each kind is addressed in a prompt. MiniMax H3 presents them as `<Picture 1>`, `<Video 1>`,
#: `<Audio 1>`, numbered per kind rather than across the whole list.
_LABELS = {
    ReferenceKind.IMAGE: "Picture",
    ReferenceKind.VIDEO: "Video",
    ReferenceKind.AUDIO: "Audio",
}


@dataclass(frozen=True)
class Reference:
    """One wired reference, with the position its prompt will address it by."""

    kind: ReferenceKind
    value: Any
    index: int  # 1-based, per kind

    @property
    def label(self) -> str:
        return f"<{_LABELS[self.kind]} {self.index}>"


@dataclass(frozen=True)
class ReferenceLimits:
    """What a model accepts. MiniMax H3 is 9 images, 3 videos, 3 audio clips, 12 in total."""

    max_images: int
    max_videos: int
    max_audio: int
    max_total: int


def collect_references(
    inputs: dict[str, list[Any]],
    *,
    image_port: str = "references",
    video_port: str = "video",
    audio_port: str = "audio",
    limits: ReferenceLimits | None = None,
) -> tuple[Reference, ...]:
    """Wired references in order: every image, then every video, then every audio clip.

    Ports are laid out in a fixed sequence rather than interleaved, because a runner receives one
    list per port and there is no cross-port wiring order left to recover. Within a port the wiring
    order is preserved exactly, which is the part a user sees and controls on the canvas.
    """
    ordered: list[Reference] = []
    for kind, port in (
        (ReferenceKind.IMAGE, image_port),
        (ReferenceKind.VIDEO, video_port),
        (ReferenceKind.AUDIO, audio_port),
    ):
        for position, value in enumerate(_wired(inputs.get(port)), start=1):
            ordered.append(Reference(kind=kind, value=value, index=position))
    if limits is not None:
        check_limits(ordered, limits)
    return tuple(ordered)


def _wired(values: Sequence[Any] | None) -> list[Any]:
    """Drop unwired slots without renumbering around them - a gap would shift every later index."""
    return [value for value in (values or []) if value is not None]


def count_by_kind(references: Sequence[Reference]) -> dict[ReferenceKind, int]:
    counts = dict.fromkeys(ReferenceKind, 0)
    for reference in references:
        counts[reference.kind] += 1
    return counts


def check_limits(references: Sequence[Reference], limits: ReferenceLimits) -> None:
    """Refuse an over-full request up front, naming how many to unwire.

    Cheaper as a graph error than as a failure part-way through a load, and far cheaper than a
    render that quietly ignored the extras.
    """
    counts = count_by_kind(references)
    for kind, allowed in (
        (ReferenceKind.IMAGE, limits.max_images),
        (ReferenceKind.VIDEO, limits.max_videos),
        (ReferenceKind.AUDIO, limits.max_audio),
    ):
        if counts[kind] > allowed:
            raise ComponentError(
                f"{counts[kind]} {kind.value} references are wired, but this model takes at most "
                f"{allowed}. Unwire {counts[kind] - allowed}."
            )
    if len(references) > limits.max_total:
        raise ComponentError(
            f"{len(references)} references are wired, but this model takes at most "
            f"{limits.max_total} in total. Unwire {len(references) - limits.max_total}."
        )


def describe(references: Sequence[Reference]) -> str:
    """The numbering as the node face shows it, for logs and error messages."""
    return ", ".join(reference.label for reference in references) or "none"
