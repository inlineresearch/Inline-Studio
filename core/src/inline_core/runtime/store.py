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
