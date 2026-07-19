"""What a node needs on disk, and who answers that question for it.

The model popup asks one thing - *"which weight files does this node need, are they here, and where
do I get the missing ones?"* - and until now only Z-Image could answer, by a hardcoded node-type
check in the Studio download layer. This module turns that into a registry keyed by node type, so a
built-in model and a community extension answer it the same way and share the whole download UI.

Providers are **torch-free and pure-filesystem**: they run on every popup open, on installs with no
ML stack, and must never fetch anything. Downloading stays exclusively the Studio downloader's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelComponent:
    """One required model component: whether it's present, and the exact file the popup fetches."""

    id: str
    label: str
    category: str  # models/ subfolder it belongs to
    present: bool
    filename: str  # the single file that lands flat in models/<category>/ (a dropdown entry)
    repo: str  # HF repo the popup downloads from
    repo_file: str  # exact repo-relative path fetched from ``repo``

    @property
    def local_path(self) -> str:
        """Where this file lives / lands, relative to the models root (flat in its category)."""
        return f"{self.category}/{self.filename}"


@runtime_checkable
class RequirementsProvider(Protocol):
    """Answers the model popup for one or more node types.

    Must be torch-free and pure-filesystem - it runs on every popup open, including on a torch-less
    install where the node itself is unavailable.
    """

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        """The components this node needs, with live presence."""
        ...

    def download_target(self, component: ModelComponent) -> Path:
        """Absolute directory the component's file lands in - under the models root, flat, never
        the hidden Hugging Face cache."""
        ...

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        """Whether the model will fit this machine, or None when it can't be sized.

        Returning None is always safe and is the right answer when the on-disk footprint isn't
        knowable - a wrong fit estimate is worse than no estimate.
        """
        ...


class RequirementsRegistry:
    """Maps a node type to the provider that answers its model requirements.

    Registration replaces by node type, matching ``graph.Registry``. Built-ins register first and
    extensions last, so the loader can detect and refuse a collision.
    """

    def __init__(self) -> None:
        self._providers: dict[str, RequirementsProvider] = {}

    def register(self, node_type: str, provider: RequirementsProvider) -> None:
        self._providers[node_type] = provider

    def unregister(self, node_type: str) -> None:
        """Drop a provider when its extension is disabled; unknown types are ignored."""
        self._providers.pop(node_type, None)

    def get(self, node_type: str) -> RequirementsProvider | None:
        return self._providers.get(node_type)

    def has(self, node_type: str) -> bool:
        return node_type in self._providers

    def node_types(self) -> list[str]:
        return sorted(self._providers)
