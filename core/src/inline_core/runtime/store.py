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

    # Concrete, not abstract: an existing store that only knows images stays valid.
    def save_video(
        self,
        run_id: str,
        node_id: str,
        frames: Any,
        params: dict[str, Any],
        fps: float = 16.0,
    ) -> Take:
        """Persist a decoded frame sequence as an immutable video take."""
        raise NotImplementedError(f"{type(self).__name__} cannot save video takes.")
