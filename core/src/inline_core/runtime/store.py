"""The take store seam: persist a decoded output as an immutable take (bytes, hash, uri).

Phase 1's implementation writes into the project's takes/ folder. Kept behind this interface so a
fleet object store swaps in without touching the executor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..takes import Take


class TakeStore(ABC):
    @abstractmethod
    def save(self, run_id: str, node_id: str, image: Any, params: dict[str, Any]) -> Take:
        """Persist a decoded image (PIL, numpy, or tensor) as an immutable take."""

    # Media beyond images arrived after the interface did, and a store is something an extension may
    # supply, so these are concrete: an image-only store keeps importing and a video model reaching
    # one gets a message naming the store rather than an AttributeError from inside a runner.
    def save_video(
        self,
        run_id: str,
        node_id: str,
        frames: Any,
        params: dict[str, Any],
        *,
        fps: float,
        audio: Any = None,
        sample_rate: int | None = None,
    ) -> Take:
        """Persist decoded frames as one immutable take, with ``audio`` muxed into the same file.

        One take is one playable file: a model that generates video and its soundtrack jointly must
        not leave a user holding a silent clip and a stray waveform.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot store video takes.")

    def save_audio(
        self, run_id: str, node_id: str, waveform: Any, params: dict[str, Any], *, sample_rate: int
    ) -> Take:
        """Persist a decoded waveform as an immutable take."""
        raise NotImplementedError(f"{type(self).__name__} cannot store audio takes.")
