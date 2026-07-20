"""Routes an extension's private dependencies to its own ``site/``.

Ownership is decided at **install** time, never here: Python checks ``sys.modules`` before
``sys.meta_path``, so a finder cannot give two extensions different versions of the same module:
the first import wins and the second extension's finder is never called. The installer therefore
assigns one owner per top-level name, and a real conflict fails the install rather than mismatching
silently.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import Distribution, MetadataPathFinder
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class OwnedModules:
    extension_id: str
    site: Path
    modules: frozenset[str]


class ExtensionFinder(importlib.abc.MetaPathFinder):
    """Maps a fixed set of top-level module names to the private site dir that owns them."""

    def __init__(self) -> None:
        self._owner: dict[str, OwnedModules] = {}

    def add(self, extension: OwnedModules) -> None:
        # setdefault, not assignment: re-pointing a module another extension already owns would
        # reintroduce the load-order nondeterminism this design exists to prevent.
        for module in extension.modules:
            self._owner.setdefault(module, extension)

    def remove(self, extension_id: str) -> None:
        self._owner = {m: p for m, p in self._owner.items() if p.extension_id != extension_id}

    def owner_of(self, module: str) -> str | None:
        extension = self._owner.get(module)
        return extension.extension_id if extension else None

    def sites(self) -> list[str]:
        return sorted({str(p.site) for p in self._owner.values()})

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if path is not None:
            return None  # submodule: the parent package's __path__ already routes it
        extension = self._owner.get(fullname.partition(".")[0])
        if extension is None:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, [str(extension.site)], target)

    def find_distributions(self, context: Any = None) -> Iterable[Distribution]:
        # Distribution discovery is a separate protocol from import; without this a vendored dep
        # that checks its own version raises PackageNotFoundError despite importing fine.
        for site in self.sites():
            yield from MetadataPathFinder().find_distributions(_context_for(context, site))


def _context_for(context: Any, site: str) -> Any:
    from importlib.metadata import DistributionFinder

    name = getattr(context, "name", None)
    return DistributionFinder.Context(name=name, path=[site])


def install_finder() -> ExtensionFinder:
    """Install (or return) the process-wide finder."""
    for entry in sys.meta_path:
        if isinstance(entry, ExtensionFinder):
            return entry
    finder = ExtensionFinder()
    # Prepended so a private dep is never shadowed by a same-named module in the cwd.
    sys.meta_path.insert(0, finder)
    return finder


def uninstall_finder() -> None:
    sys.meta_path[:] = [e for e in sys.meta_path if not isinstance(e, ExtensionFinder)]
