"""``extensions/state.json``: which extensions are installed, active, and enabled.

Source of truth for *intent*; the version directories are the source of truth for *content*. Boot
reconciles them - an extension whose recorded version directory vanished is reported broken.

``import_owners`` is the materialized dependency resolution (one extension per top-level
derivable from the lockfiles but stored so boot is a pure file read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .manifest import JsonObject, as_array, as_object
from .paths import ExtensionsRoot, write_atomic

STATE_SCHEMA = 1


@dataclass
class ExtensionState:
    current: str = ""
    enabled: bool = True
    #: node type -> enabled. Absent means "use the manifest's defaultEnabled".
    nodes: dict[str, bool] = field(default_factory=lambda: {})
    #: Security findings the user explicitly accepted, by rule id.
    consents: list[str] = field(default_factory=lambda: [])
    #: True when the user chose this version explicitly; update checks leave it alone.
    pinned: bool = False

    def node_enabled(self, node_type: str, *, default: bool) -> bool:
        return self.nodes.get(node_type, default)


class StateStore:
    """Reads and writes ``state.json``. Every mutation persists immediately; no explicit save,
    mirroring the project DB."""

    def __init__(self, paths: ExtensionsRoot) -> None:
        self._paths = paths
        self._extensions: dict[str, ExtensionState] = {}
        self._import_owners: dict[str, str] = {}
        self.reload()

    # --- reading ---------------------------------------------------------------------------------

    def reload(self) -> None:
        self._extensions = {}
        self._import_owners = {}
        try:
            data = as_object(json.loads(self._paths.state.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return  # absent or corrupt: start empty rather than refusing to boot
        if data is None or data.get("schema") != STATE_SCHEMA:
            return  # a future/unknown schema: ignore rather than misread it
        extensions = as_object(data.get("extensions"))
        if extensions is not None:
            for extension_id, raw_entry in extensions.items():
                entry = as_object(raw_entry)
                if entry is not None:
                    self._extensions[extension_id] = _extension_state(entry)
        owners = as_object(data.get("importOwners"))
        if owners is not None:
            self._import_owners = {k: v for k, v in owners.items() if isinstance(v, str)}

    def extension(self, extension_id: str) -> ExtensionState | None:
        return self._extensions.get(extension_id)

    def extensions(self) -> dict[str, ExtensionState]:
        return dict(self._extensions)

    def import_owners(self) -> dict[str, str]:
        return dict(self._import_owners)

    def owner_of(self, module: str) -> str | None:
        return self._import_owners.get(module)

    # --- writing ---------------------------------------------------------------------------------

    def activate(
        self,
        extension_id: str,
        *,
        version: str,
        owns: list[str],
        consents: list[str] | None = None,
        nodes: dict[str, bool] | None = None,
    ) -> None:
        """Record an activation. ``owns`` is the Python import names this extension resolves
        privately; ``nodes`` is which of its nodes are enabled. Ownership it no longer
        needs is released, so a stale claim can't block another extension."""
        state = self._extensions.setdefault(extension_id, ExtensionState())
        state.current = version
        state.enabled = True
        if consents is not None:
            state.consents = list(consents)
        if nodes is not None:
            state.nodes = dict(nodes)
        self._claim_imports(extension_id, owns)
        self._save()

    def set_enabled(self, extension_id: str, enabled: bool) -> None:
        state = self._extensions.get(extension_id)
        if state is None:
            return
        state.enabled = enabled
        self._save()

    def set_node_enabled(self, extension_id: str, node_type: str, enabled: bool) -> None:
        state = self._extensions.get(extension_id)
        if state is None:
            return
        state.nodes[node_type] = enabled
        self._save()

    def set_current(self, extension_id: str, version: str, *, pinned: bool = True) -> None:
        """Point an extension at an installed version (the rollback / version-switch path)."""
        state = self._extensions.get(extension_id)
        if state is None:
            return
        state.current = version
        state.pinned = pinned
        self._save()

    def remove(self, extension_id: str) -> None:
        self._extensions.pop(extension_id, None)
        self._import_owners = {m: o for m, o in self._import_owners.items() if o != extension_id}
        self._save()

    def _claim_imports(self, extension_id: str, owns: list[str]) -> None:
        wanted = set(owns)
        self._import_owners = {
            name: owner
            for name, owner in self._import_owners.items()
            if owner != extension_id or name in wanted
        }
        for name in wanted:
            self._import_owners[name] = extension_id

    def _save(self) -> None:
        payload = {
            "schema": STATE_SCHEMA,
            "extensions": {
                extension_id: {
                    "current": s.current,
                    "enabled": s.enabled,
                    "nodes": s.nodes,
                    "consents": s.consents,
                    "pinned": s.pinned,
                }
                for extension_id, s in sorted(self._extensions.items())
            },
            "importOwners": dict(sorted(self._import_owners.items())),
        }
        write_atomic(self._paths.state, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _extension_state(entry: JsonObject) -> ExtensionState:
    nodes = as_object(entry.get("nodes")) or {}
    consents = as_array(entry.get("consents")) or []
    return ExtensionState(
        current=str(entry.get("current", "")),
        enabled=bool(entry.get("enabled", True)),
        nodes={key: bool(value) for key, value in nodes.items()},
        consents=[c for c in consents if isinstance(c, str)],
        pinned=bool(entry.get("pinned", False)),
    )
