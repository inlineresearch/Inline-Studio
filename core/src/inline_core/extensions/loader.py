"""Boot-time extension loading: read state, route private imports, register modules.

Failure is contained per module and surfaced on ``LoadedExtension``, never raised - hence the broad
``except Exception``, the same plugin-boundary pattern used by RPC dispatch and the uploader.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..device.policy import DevicePolicy
from ..graph.registry import Registry
from ..models.requirements import RequirementsRegistry
from ..runtime.store import TakeStore
from .api import ExtensionRegistrar
from .importer import ExtensionFinder, OwnedModules, install_finder
from .manifest import Manifest, ManifestError, as_array, as_object, load_manifest
from .models import ManifestRequirements
from .paths import ExtensionsRoot, VersionPaths
from .state import StateStore

log = logging.getLogger(__name__)


@dataclass
class LoadedExtension:
    extension_id: str
    version: str
    name: str = ""
    enabled: bool = True
    #: Node types actually registered (the enabled ones).
    node_types: tuple[str, ...] = ()
    #: Declared but switched off by the user - toggling one on needs no restart.
    skipped: tuple[str, ...] = ()
    #: Set when the extension failed (bad manifest, missing files, incompatible Core, bad import).
    error: str | None = None


def load_extensions(
    registry: Registry,
    store: TakeStore,
    policy: DevicePolicy,
    *,
    requirements: RequirementsRegistry | None = None,
    rpc: Any = None,
    events: Any = None,
    root: Path | None = None,
    core_version: str | None = None,
) -> list[LoadedExtension]:
    """Load every enabled extension recorded in ``state.json``."""
    paths = ExtensionsRoot.resolve(root)
    if not paths.root.is_dir():
        return []
    paths.ensure_dirs()
    state = StateStore(paths)
    reqs = requirements if requirements is not None else RequirementsRegistry()
    finder = install_finder()

    loaded: list[LoadedExtension] = []
    for extension_id, pack_state in sorted(state.extensions().items()):
        if not pack_state.enabled:
            loaded.append(LoadedExtension(extension_id, pack_state.current, enabled=False))
            continue
        loaded.append(
            _load_pack(
                extension_id,
                paths,
                state,
                registry,
                store,
                policy,
                reqs,
                finder,
                rpc,
                events,
                core_version,
            )
        )
    _empty_trash(paths)
    return loaded


def _load_pack(
    extension_id: str,
    paths: ExtensionsRoot,
    state: StateStore,
    registry: Registry,
    store: TakeStore,
    policy: DevicePolicy,
    requirements: RequirementsRegistry,
    finder: ExtensionFinder,
    rpc: Any,
    events: Any,
    core_version: str | None,
) -> LoadedExtension:
    extension_state = state.extension(extension_id)
    version = extension_state.current if extension_state else ""
    extension_paths = extension_paths_for(paths, extension_id)
    current = extension_paths.current()
    if current is None:
        return LoadedExtension(
            extension_id, version, error="installed files are missing; reinstall the extension"
        )

    version_paths = extension_paths.version(current)
    try:
        manifest = _manifest_of(version_paths, extension_id)
    except ManifestError as error:
        return LoadedExtension(extension_id, current, error=f"invalid manifest: {error}")

    if core_version is not None and not _compatible(manifest, core_version):
        return LoadedExtension(
            extension_id,
            current,
            name=manifest.name,
            error=f"requires Inline Core {manifest.core_compat} (running {core_version})",
        )

    return load_extension_into(
        manifest,
        version_paths,
        registry=registry,
        store=store,
        policy=policy,
        requirements=requirements,
        state=state,
        rpc=rpc,
        events=events,
        finder=finder,
    )


def load_extension_into(
    manifest: Manifest,
    version_paths: VersionPaths,
    *,
    registry: Registry,
    store: TakeStore,
    policy: DevicePolicy,
    requirements: RequirementsRegistry,
    state: StateStore,
    rpc: Any = None,
    events: Any = None,
    finder: ExtensionFinder | None = None,
) -> LoadedExtension:
    """Import an extension once and register its enabled nodes. Shared by boot and by a fresh
    install, so a first-time install goes live without a restart."""
    extension_state = state.extension(manifest.id)
    enabled = [
        node.type
        for node in manifest.nodes
        if (
            extension_state.node_enabled(node.type, default=node.default_enabled)
            if extension_state
            else node.default_enabled
        )
    ]
    # Route private dependencies before importing any of the extension's code.
    (finder or install_finder()).add(_owned_modules(manifest.id, version_paths))
    _ensure_on_path(version_paths)

    registrar = ExtensionRegistrar(
        registry,
        manifest.id,
        store=store,
        policy=policy,
        requirements=requirements,
        data_root=version_paths.root.parent.parent / "data",
        declared_nodes=manifest.node_types(),
        enabled_nodes=enabled,
        rpc=rpc,
        events=events,
    )
    try:
        resolve_entry(manifest.entry)(registrar)
        # Manifest-declared weights need no extension code of their own.
        for node in manifest.nodes:
            if node.models and node.type in registrar.registered_nodes:
                requirements.register(node.type, ManifestRequirements(node.models))
    except Exception as error:  # noqa: BLE001 - plugin boundary; must not kill boot
        log.warning("extension %s failed to load: %s", manifest.id, error)
        _rollback(registry, rpc, requirements, registrar)
        return LoadedExtension(
            manifest.id, version_paths.root.name, name=manifest.name, error=str(error)
        )
    return LoadedExtension(
        manifest.id,
        version_paths.root.name,
        name=manifest.name,
        node_types=tuple(registrar.registered_nodes),
        skipped=tuple(registrar.skipped_nodes),
    )


def _rollback(
    registry: Registry,
    rpc: Any,
    requirements: RequirementsRegistry,
    registrar: ExtensionRegistrar,
) -> None:
    """Undo a partially-registered extension, so a failure halfway through ``register()``
    cannot leave half its nodes live."""
    for node_type in registrar.registered_nodes:
        registry.unregister(node_type)
        requirements.unregister(node_type)
    if rpc is not None:
        for channel in registrar.registered_channels:
            rpc.unregister(channel)


def resolve_entry(entry: str) -> Any:
    module_name, _, attr = entry.partition(":")
    module = importlib.import_module(module_name)
    target = getattr(module, attr, None)
    if not callable(target):
        raise ImportError(f"entry point {entry!r} is not callable")
    return target


def _manifest_of(version_paths: VersionPaths, extension_id: str) -> Manifest:
    """Prefer the normalized copy written at install time; fall back to the repo's own file."""
    if version_paths.manifest.is_file():
        try:
            raw = json.loads(version_paths.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError([f"$: could not read the installed manifest ({error})"]) from error
        from .manifest import parse_manifest

        return parse_manifest(raw, expect_id=extension_id)
    return load_manifest(version_paths.source, expect_id=extension_id)


def _owned_modules(extension_id: str, version_paths: VersionPaths) -> OwnedModules:
    owned: list[str] = []
    try:
        raw = as_object(json.loads(version_paths.owned_modules.read_text(encoding="utf-8")))
        top = as_array(raw.get("topLevel")) if raw is not None else None
        owned = [m for m in (top or []) if isinstance(m, str)]
    except (OSError, json.JSONDecodeError):
        owned = []  # no private deps recorded: the extension resolves entirely against the host
    return OwnedModules(extension_id=extension_id, site=version_paths.site,
    modules=frozenset(owned))


def _ensure_on_path(version_paths: VersionPaths) -> None:
    """The extension's own code lives in ``source/python/`` and is imported by its mandatory unique
    top-level name, so a plain ``sys.path`` entry is safe."""
    python_root = str((version_paths.source / "python").resolve())
    if python_root not in sys.path:
        sys.path.insert(0, python_root)


def _compatible(manifest: Manifest, core_version: str) -> bool:
    """Best-effort PEP 440 check. Without ``packaging`` installed we allow the load rather than
    locking every extension out - the manifest range is advisory, not a security control."""
    spec = manifest.core_compat.strip()
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:
        return True
    try:
        return Version(core_version) in SpecifierSet(spec)
    except Exception:  # noqa: BLE001 - a malformed range must not block boot
        return True


def extension_paths_for(paths: ExtensionsRoot, extension_id: str) -> Any:
    return paths.extension(extension_id)


def _empty_trash(paths: ExtensionsRoot) -> None:
    for extension_id in paths.installed_extensions():
        try:
            paths.extension(extension_id).empty_trash()
        except OSError:
            continue
