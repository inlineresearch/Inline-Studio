"""Immutable outputs (takes) and input references (asset refs). See docs/contract.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .media import MediaKind


@dataclass(frozen=True)
class AssetRef:
    """Input bytes: a content-addressed upload (`asset`) or a local file Core can read (`path`)."""

    ref: Literal["asset", "path"]
    id: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class Take:
    """One immutable render produced by a node."""

    id: str
    run_id: str
    node_id: str
    kind: MediaKind
    uri: str
    hash: str
    params: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
