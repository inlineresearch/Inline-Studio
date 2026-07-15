"""The media kinds a take or a model output can be."""

from __future__ import annotations

from enum import Enum


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
