"""The install state machine.

Every phase writes only inside a staging directory. Nothing outside it is touched until ACTIVATE,
so a failure at any point has nothing to undo - that is the entire rollback story.

REGISTER imports into a *scratch* registry, so an extension that raises on import or collides on a
node type is caught before the live registry is mutated.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..device.policy import DevicePolicy
from ..graph.registry import Registry
from ..models.requirements import RequirementsRegistry
from ..runtime.store import TakeStore
from .api import ExtensionRegistrar
from .constraints import Conflict
from .fetch import FetchError, fetch, latest_tag, remote_sha
from .loader import resolve_entry
from .manifest import (
    JsonObject,
    Manifest,
    ManifestError,
    as_object,
    load_manifest,
    parse_manifest,
)
from .paths import ExtensionsRoot, VersionPaths, version_dirname, write_atomic
from .resolve import Resolution, ResolutionError, resolve_and_install
from .scanner import ScanReport, scan
from .state import StateStore
from .tools import ToolMissing, missing_tools, require_tools


class Phase(StrEnum):
    FETCH = "fetch"
    VALIDATE = "validate"
    SCAN = "scan"
    PREFLIGHT = "preflight"
    RESOLVE = "resolve"
    INSTALL = "install"
    LOCK = "lock"
    REGISTER = "register"
    ACTIVATE = "activate"
    DONE = "done"


class InstallError(RuntimeError):
    """A failed install. Carries the structured detail the dialog needs to explain itself."""

    def __init__(
        self,
        message: str,
        *,
        phase: Phase,
        conflicts: list[Conflict] | None = None,
        report: ScanReport | None = None,
        restart_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.conflicts = conflicts or []
        self.report = report
        self.restart_required = restart_required

    def to_json(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "phase": self.phase.value,
            "conflicts": [c.to_json() for c in self.conflicts],
            "scan": self.report.to_json() if self.report else None,
            "restartRequired": self.restart_required,
        }


@dataclass(frozen=True)
class InstallRequest:
    source: str
    ref: str = "main"
    #: Scan rules the user accepted. A HIGH/MEDIUM finding not listed here pauses the install.
    consents: tuple[str, ...] = ()


@dataclass
class InstallResult:
    extension_id: str
    version: str
    name: str = ""
    node_types: list[str] = field(default_factory=lambda: [])
    scan: ScanReport | None = None
    #: True when the extension was already imported this session, so the new code is not live yet.
    restart_required: bool = False
    #: Set when the install paused for consent; nothing was activated.
    needs_consent: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "extensionId": self.extension_id,
            "version": self.version,
            "name": self.name,
            "nodeTypes": self.node_types,
            "scan": self.scan.to_json() if self.scan else None,
            "restartRequired": self.restart_required,
            "needsConsent": self.needs_consent,
        }


class Installer:
    """Runs installs and the lifecycle operations around them."""

    def __init__(
        self,
        registry: Registry,
        store: TakeStore,
        policy: DevicePolicy,
        *,
        requirements: RequirementsRegistry,
        paths: ExtensionsRoot | None = None,
        rpc: Any = None,
        events: Any = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._policy = policy
        self._requirements = requirements
        self._paths = paths or ExtensionsRoot.resolve()
        self._rpc = rpc
        self._events = events
        self._paths.ensure_dirs()
        self._state = StateStore(self._paths)
        #: Extensions imported this session. Re-installing one of these needs a restart to take
        #: effect.
        self._imported: set[str] = set()

    @property
    def state(self) -> StateStore:
        return self._state

    @property
    def paths(self) -> ExtensionsRoot:
        """The root this installer works in. Callers must use this rather than resolving from the
        environment, or they silently read and write a different directory."""
        return self._paths

    # --- install ----------------------------------------------------------------------------------

    async def install(self, request: InstallRequest) -> InstallResult:
        """Run the pipeline off the event loop; git and uv both block."""
        loop = asyncio.get_running_loop()
        return await asyncio.to_thread(self._install, request, loop)

    def _install(self, request: InstallRequest, loop: Any = None) -> InstallResult:
        if missing := missing_tools():
            raise InstallError(
                f"installing extensions needs {' and '.join(missing)}", phase=Phase.FETCH
            )
        require_tools()

        token = uuid.uuid4().hex[:12]
        staging = self._paths.new_staging(token)
        try:
            return self._run_phases(request, staging, loop)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _run_phases(self, request: InstallRequest, staging: Path, loop: Any) -> InstallResult:
        source = staging / "source"

        self._emit(loop, Phase.FETCH, 0.05, "Downloading…")
        ref = self._resolve_ref(request.source, request.ref)
        try:
            fetched = fetch(
                request.source,
                ref,
                mirror=self._paths.git_mirror(_mirror_name(request.source)),
                dest=source,
            )
        except (FetchError, ToolMissing) as error:
            raise InstallError(str(error), phase=Phase.FETCH) from error

        self._emit(loop, Phase.VALIDATE, 0.15, "Checking the manifest…")
        try:
            manifest = load_manifest(source)
        except ManifestError as error:
            raise InstallError(str(error), phase=Phase.VALIDATE) from error

        self._emit(loop, Phase.SCAN, 0.25, "Reviewing the code…")
        report = scan(source, manifest)
        if report.blocked:
            raise InstallError(
                _blocked_message(report), phase=Phase.SCAN, report=report
            )
        outstanding = set(report.consent_rules()) - set(request.consents)
        if outstanding:
            # Pause, don't fail: the dialog shows the report and re-calls with consents.
            return InstallResult(
                extension_id=manifest.id,
                version=manifest.version,
                name=manifest.name,
                scan=report,
                needs_consent=True,
            )

        self._emit(loop, Phase.PREFLIGHT, 0.35, "Checking for conflicts…")
        self._preflight(manifest)

        version = version_dirname(manifest.version, fetched.sha)
        target = self._paths.extension(manifest.id).version(version)

        self._emit(loop, Phase.RESOLVE, 0.45, "Resolving dependencies…")
        resolution = self._resolve(manifest, staging)

        self._emit(loop, Phase.LOCK, 0.75, "Recording the install…")
        self._write_metadata(
            staging, manifest, fetched.sha, fetched.ref, request.source, report, resolution
        )

        self._emit(loop, Phase.REGISTER, 0.85, "Loading nodes…")
        titles = self._register_scratch(manifest, staging)

        self._emit(loop, Phase.ACTIVATE, 0.95, "Activating…")
        result = self._activate(manifest, staging, target, version, resolution, report, titles)

        self._emit(loop, Phase.DONE, 1.0, "Installed")
        self._broadcast(loop, "events:extensionInstallDone", result.to_json())
        return result

    def _resolve_ref(self, source: str, ref: str) -> str:
        """Turn a floating request into a concrete tag.

        A registry listing names the repository, not a version: authors publish by tagging, so
        ``latest`` resolves to the newest release tag here. An explicit ref is used as given.
        """
        if ref and ref != "latest":
            return ref
        newest = latest_tag(source)
        if newest is None:
            raise InstallError(
                "no release tag found in that repository; tag a release (for example v1.0.0) "
                "or install a branch by name",
                phase=Phase.FETCH,
            )
        return newest

    def _preflight(self, manifest: Manifest) -> None:
        """Catch collisions before any extension code runs."""
        for node_type in manifest.node_types():
            if not self._registry.has(node_type):
                continue
            source = self._registry.get(node_type).source
            # Reinstalling or upgrading the same extension is fine; another owner is not.
            if source != f"ext:{manifest.id}":
                owner = "Inline Core" if source == "builtin" else source
                raise InstallError(
                    f"node type {node_type!r} is already provided by {owner}",
                    phase=Phase.PREFLIGHT,
                )
        for module, owner in self._state.import_owners().items():
            if owner != manifest.id and module in {r.split("=")[0] for r in manifest.requirements}:
                raise InstallError(
                    f"{module!r} is already provided by the {owner!r} extension; "
                    "disable it or align the versions",
                    phase=Phase.PREFLIGHT,
                )

    def _resolve(self, manifest: Manifest, staging: Path) -> Resolution:
        if _prebuilt_for(manifest) is not None:
            # Seam for the Node-Packer path: a matching tarball skips resolution entirely.
            raise InstallError(
                "prebuilt extension packages are not supported yet; "
                "install from source or ask the author to publish a source install",
                phase=Phase.RESOLVE,
            )
        try:
            return resolve_and_install(
                manifest.requirements,
                site=staging / "site",
                lock_dir=staging / "lock",
                constraints_path=self._paths.constraints,
                log=staging / "lock" / "install.log",
            )
        except ResolutionError as error:
            raise InstallError(
                str(error), phase=Phase.RESOLVE, conflicts=error.conflicts
            ) from error

    def _write_metadata(
        self,
        staging: Path,
        manifest: Manifest,
        sha: str,
        ref: str,
        source: str,
        report: ScanReport,
        resolution: Resolution,
    ) -> None:
        raw = (staging / "source" / "inline-extension.json").read_text(encoding="utf-8")
        write_atomic(staging / "manifest.json", raw)
        write_atomic(staging / "scan.json", json.dumps(report.to_json(), indent=2))
        write_atomic(
            staging / "lock" / "owned-modules.json",
            json.dumps(resolution.to_json(), indent=2),
        )
        write_atomic(
            staging / "receipt.json",
            json.dumps(
                {
                    "sha": sha,
                    "ref": ref,
                    "source": source,
                    "version": manifest.version,
                    "python": sys.version.split()[0],
                    "consents": report.consent_rules(),
                },
                indent=2,
            ),
        )

    def _register_scratch(self, manifest: Manifest, staging: Path) -> dict[str, str]:
        """Import and register into a throwaway registry first, so a broken extension never
        touches the live one. Import failures leave modules in sys.modules, hence restart_required.

        Every declared node is registered here, including default-off ones: an extension that only
        breaks once a node is switched on would be a nasty surprise later.
        """
        scratch = Registry()
        python_root = str((staging / "source" / "python").resolve())
        sys.path.insert(0, python_root)
        try:
            registrar = ExtensionRegistrar(
                scratch,
                manifest.id,
                store=self._store,
                policy=self._policy,
                requirements=RequirementsRegistry(),
                data_root=staging / "data",
                declared_nodes=manifest.node_types(),
            )
            try:
                resolve_entry(manifest.entry)(registrar)
            except Exception as error:  # noqa: BLE001 - plugin boundary
                raise InstallError(
                    str(error), phase=Phase.REGISTER, restart_required=True
                ) from error
            missing = set(manifest.node_types()) - set(registrar.registered_nodes)
            if missing:
                raise InstallError(
                    f"the manifest declares {', '.join(sorted(missing))} but register() did not "
                    "provide them",
                    phase=Phase.REGISTER,
                    restart_required=True,
                )
            # Titles come from the descriptor, so a node the user has switched off still shows a
            # readable name in the dialog rather than its raw type.
            return {t: scratch.get(t).title for t in registrar.registered_nodes}
        finally:
            sys.path[:] = [p for p in sys.path if p != python_root]

    def _activate(
        self,
        manifest: Manifest,
        staging: Path,
        target: VersionPaths,
        version: str,
        resolution: Resolution,
        report: ScanReport,
        titles: dict[str, str],
    ) -> InstallResult:
        extension = self._paths.extension(manifest.id)
        extension.versions.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target.root, ignore_errors=True)
        shutil.move(str(staging), str(target.root))

        previous = extension.current()
        extension.set_current(version)
        self._state.activate(
            manifest.id,
            version=version,
            owns=resolution.modules,
            consents=report.consent_rules(),
            nodes={n.type: n.default_enabled for n in manifest.nodes},
        )
        extension.prune()

        restart = manifest.id in self._imported
        if not restart:
            self._load_now(manifest, target)
        live = {n.type for n in manifest.nodes if n.default_enabled}
        write_atomic(target.titles, json.dumps(titles, indent=2))
        return InstallResult(
            extension_id=manifest.id,
            version=version,
            name=manifest.name,
            node_types=[n for n in titles if n in live],
            scan=report,
            restart_required=restart or previous is not None,
        )

    def _load_now(self, manifest: Manifest, target: VersionPaths) -> None:
        """Register into the live registry, so a first install needs no restart."""
        from .loader import load_extension_into

        load_extension_into(
            manifest,
            target,
            registry=self._registry,
            store=self._store,
            policy=self._policy,
            requirements=self._requirements,
            state=self._state,
            rpc=self._rpc,
            events=self._events,
        )
        self._imported.add(manifest.id)

    # --- lifecycle --------------------------------------------------------------------------------

    def list_packs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for extension_id in self._paths.installed_extensions():
            extension = self._paths.extension(extension_id)
            current = extension.current()
            extension_state = self._state.extension(extension_id)
            manifest = self._manifest_of(extension_id, current)
            titles = self._titles_of(extension_id, current)
            receipt = self._receipt_of(extension_id, current)
            out.append(
                {
                    "extensionId": extension_id,
                    "name": manifest.name if manifest else extension_id,
                    "description": manifest.description if manifest else "",
                    "version": manifest.version if manifest else "",
                    "installed": current or "",
                    "enabled": extension_state.enabled if extension_state else False,
                    "versions": extension.installed_versions(),
                    "homepage": manifest.homepage if manifest else "",
                    "license": manifest.license if manifest else "",
                    "repo": str(receipt.get("source", "")),
                    "ref": str(receipt.get("ref", "")),
                    "sha": str(receipt.get("sha", ""))[:7],
                    "nodes": [
                        {
                            "type": node.type,
                            "title": titles.get(node.type, node.type),
                            "enabled": extension_state.node_enabled(
                                node.type, default=node.default_enabled
                            )
                            if extension_state
                            else False,
                        }
                        for node in (manifest.nodes if manifest else ())
                    ],
                }
            )
        return out

    def check_updates(self) -> list[dict[str, Any]]:
        """Ask each extension's origin what its ref points at now, so the card can show drift.

        Network-bound and best-effort, hence separate from ``list``: the dialog opens instantly and
        the badges fill in when this returns. An unreachable origin reports ``checked: false``
        rather than claiming the extension is up to date.
        """
        out: list[dict[str, Any]] = []
        for extension_id in self._paths.installed_extensions():
            current = self._paths.extension(extension_id).current()
            receipt = self._receipt_of(extension_id, current)
            source, ref = str(receipt.get("source", "")), str(receipt.get("ref", ""))
            installed = str(receipt.get("sha", ""))
            if not source or not ref or not installed:
                continue
            newest = latest_tag(source)
            # Two ways to be out of date: a newer tag exists, or the installed tag itself moved.
            upstream = remote_sha(source, newest or ref)
            behind = bool(
                (newest and newest != ref) or (upstream and upstream != installed)
            )
            out.append(
                {
                    "extensionId": extension_id,
                    "ref": ref,
                    "latestTag": newest,
                    "installedSha": installed[:7],
                    "remoteSha": upstream[:7] if upstream else None,
                    "behind": behind,
                    "checked": upstream is not None or newest is not None,
                }
            )
        return out

    def set_enabled(self, extension_id: str, enabled: bool) -> dict[str, Any]:
        self._state.set_enabled(extension_id, enabled)
        if not enabled:
            self._unload(extension_id)
            return {"extensionId": extension_id, "enabled": False, "restartRequired": False}
        restart = self._load_enabled(extension_id)
        return {"extensionId": extension_id, "enabled": True, "restartRequired": restart}

    def set_node_enabled(self, extension_id: str, node_type: str, enabled: bool) -> dict[str, Any]:
        """Toggle one node. Never needs a restart: the extension's code is already imported, so
        this only adds or removes a registry entry."""
        self._state.set_node_enabled(extension_id, node_type, enabled)
        if enabled:
            self._load_enabled(extension_id)
        else:
            self._registry.unregister(node_type)
            self._requirements.unregister(node_type)
        return {
            "extensionId": extension_id,
            "nodeType": node_type,
            "enabled": enabled,
            "restartRequired": False,
        }

    def _load_enabled(self, extension_id: str) -> bool:
        """Register whatever is enabled but not yet live. Returns True if a restart is needed."""
        extension = self._paths.extension(extension_id)
        current = extension.current()
        if current is None:
            return False
        version_paths = extension.version(current)
        manifest = self._manifest_of(extension_id, current)
        if manifest is None:
            return False
        from .loader import load_extension_into

        loaded = load_extension_into(
            manifest,
            version_paths,
            registry=self._registry,
            store=self._store,
            policy=self._policy,
            requirements=self._requirements,
            state=self._state,
            rpc=self._rpc,
            events=self._events,
        )
        self._imported.add(extension_id)
        return loaded.error is not None

    def switch_version(self, extension_id: str, version: str) -> dict[str, Any]:
        """Roll back or forward to an already-installed version. Always needs a restart: Python
        cannot unload the modules already imported from the old one."""
        extension = self._paths.extension(extension_id)
        if version not in extension.installed_versions():
            raise InstallError(f"version {version!r} is not installed", phase=Phase.ACTIVATE)
        extension.set_current(version)
        self._state.set_current(extension_id, version)
        return {"extensionId": extension_id, "version": version, "restartRequired": True}

    def uninstall(self, extension_id: str) -> dict[str, Any]:
        self._unload(extension_id)
        self._state.remove(extension_id)
        shutil.rmtree(self._paths.extension(extension_id).root, ignore_errors=True)
        return {"extensionId": extension_id, "restartRequired": extension_id in self._imported}

    def _unload(self, extension_id: str) -> None:
        """Drop every node and channel this extension registered. Its Python modules stay in
        sys.modules, unreachable but harmless."""
        source = f"ext:{extension_id}"
        for descriptor in list(self._registry.descriptors()):
            if descriptor.source == source:
                self._registry.unregister(descriptor.type)
                self._requirements.unregister(descriptor.type)
        if self._rpc is not None:
            for channel in list(getattr(self._rpc, "_handlers", {})):
                if channel.startswith(f"ext:{extension_id}:"):
                    self._rpc.unregister(channel)

    def _receipt_of(self, extension_id: str, version: str | None) -> JsonObject:
        """Install provenance: the source URL, the ref asked for, and the commit it resolved to."""
        if not version:
            return {}
        try:
            raw = as_object(
                json.loads(
                    self._paths.extension(extension_id).version(version).receipt.read_text("utf-8")
                )
            )
        except (OSError, json.JSONDecodeError):
            return {}
        return raw or {}

    def _titles_of(self, extension_id: str, version: str | None) -> dict[str, str]:
        if not version:
            return {}
        try:
            raw = as_object(
                json.loads(
                    self._paths.extension(extension_id).version(version).titles.read_text("utf-8")
                )
            )
        except (OSError, json.JSONDecodeError):
            return {}
        return {key: str(value) for key, value in raw.items()} if raw is not None else {}

    def _manifest_of(self, extension_id: str, version: str | None) -> Manifest | None:
        if not version:
            return None
        path = self._paths.extension(extension_id).version(version).manifest
        try:
            return parse_manifest(json.loads(path.read_text(encoding="utf-8")),
            expect_id=extension_id)
        except (OSError, json.JSONDecodeError, ManifestError):
            return None

    # --- events -----------------------------------------------------------------------------------

    def _emit(self, loop: Any, phase: Phase, fraction: float, status: str) -> None:
        self._broadcast(
            loop,
            "events:extensionInstallProgress",
            {"phase": phase.value, "fraction": fraction, "status": status},
        )

    def _broadcast(self, loop: Any, channel: str, payload: Any) -> None:
        if self._events is None:
            return
        if loop is None:
            self._events.broadcast(channel, payload)
            return
        # EventBroadcaster is not thread-safe and we run off the loop.
        loop.call_soon_threadsafe(self._events.broadcast, channel, payload)


def _mirror_name(url: str) -> str:
    """A filesystem-safe cache key for a repo URL."""
    cleaned = url.rstrip("/").removesuffix(".git")
    tail = cleaned.rsplit("/", 2)[-2:]
    return "-".join(part.replace(":", "-") for part in tail if part) or "repo"


def _prebuilt_for(manifest: Manifest) -> object | None:
    """Match a prebuilt artifact to this platform. Deferred: the seam exists, the fetch does not."""
    if not manifest.prebuilt:
        return None
    import platform as platform_mod

    machine = platform_mod.machine().lower()
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(
        platform_mod.system(), ""
    )
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for built in manifest.prebuilt:
        if built.python == tag and built.platform.lower() in {
            f"{system}-{machine}",
            f"{system}-{'x86_64' if machine == 'amd64' else machine}",
        }:
            return built
    return None


def _blocked_message(report: ScanReport) -> str:
    from .scanner import Severity

    critical = report.by_severity(Severity.CRITICAL)
    head = critical[0].message if critical else "the security scan blocked this extension"
    extra = f" (and {len(critical) - 1} more)" if len(critical) > 1 else ""
    return head + extra
