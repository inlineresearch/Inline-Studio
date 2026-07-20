"""The ``ext:manage:*`` RPC channels behind the Extensions dialog.

``ext:manage:`` is reserved; extension-provided channels are ``ext:<extension>:<method>`` and the
registrar rejects anything else, so an extension can never register here.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import registry_url
from .install import Installer, InstallError, InstallRequest
from .manifest import JsonObject, as_array, as_object
from .paths import ExtensionsRoot, write_atomic
from .tools import can_install, tool_status

CHANNEL_PREFIX = "ext:manage:"


def register_extension_handlers(rpc: Any, installer: Installer) -> None:
    def reg(method: str, fn: Callable[..., Any]) -> None:
        async def handler(args: list[Any]) -> Any:
            result = fn(*args)
            if inspect.isawaitable(result):
                result = await result
            return result

        rpc.register(CHANNEL_PREFIX + method, handler)

    async def install(source: str, ref: str = "main", consents: object = None) -> dict[str, Any]:
        accepted = tuple(c for c in (as_array(consents) or []) if isinstance(c, str))
        try:
            result = await installer.install(
                InstallRequest(source=source, ref=ref, consents=accepted)
            )
        except InstallError as error:
            # Structured, not a bare string: the dialog renders conflicts and scan findings.
            return {"ok": False, **error.to_json()}
        return {"ok": True, **result.to_json()}

    def status() -> dict[str, Any]:
        return {
            "canInstall": can_install(),
            "tools": [t.to_json() for t in tool_status()],
            "extensions": installer.list_packs(),
        }

    def versions(extension_id: str) -> dict[str, Any]:
        extension = installer.paths.extension(extension_id)
        return {
            "extensionId": extension_id,
            "current": extension.current() or "",
            "versions": extension.installed_versions(),
        }

    def registry_index(refresh: bool = False) -> dict[str, Any]:
        return _registry_index(installer.paths, refresh=refresh)

    reg("list", installer.list_packs)
    reg("status", status)
    reg("install", install)
    reg("uninstall", installer.uninstall)
    reg("setEnabled", installer.set_enabled)
    reg("setNodeEnabled", installer.set_node_enabled)
    reg("versions", versions)
    reg("switchVersion", installer.switch_version)
    reg("checkUpdates", installer.check_updates)
    reg("registryIndex", registry_index)


def _registry_index(paths: ExtensionsRoot, *, refresh: bool) -> dict[str, Any]:
    """The published extension list, from ``config.registry_url()``.

    Cached on disk with its ETag: a refresh is a conditional GET, and an unreachable registry
    degrades to the cached entries marked ``stale`` rather than an empty dialog.
    """
    cached, etag = _read_cache(paths)
    if not refresh and cached is not None:
        return {"entries": cached, "stale": False}

    fetched = _fetch_index(etag if cached is not None else None)
    if fetched is None:
        return {"entries": cached or [], "stale": True}
    if fetched.unchanged:
        return {"entries": cached or [], "stale": False}

    paths.cache.mkdir(parents=True, exist_ok=True)
    write_atomic(
        paths.registry_index,
        json.dumps({"entries": fetched.entries, "etag": fetched.etag}, indent=2),
    )
    return {"entries": fetched.entries, "stale": False}


def _read_cache(paths: ExtensionsRoot) -> tuple[list[JsonObject] | None, str | None]:
    try:
        raw = as_object(json.loads(paths.registry_index.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None, None
    if raw is None:
        return None, None
    tag = raw.get("etag")
    return _entries(raw.get("entries")), tag if isinstance(tag, str) else None


@dataclass
class _Fetched:
    entries: list[JsonObject]
    etag: str | None
    #: True on a 304: the cache is current and must not be overwritten with an empty body.
    unchanged: bool = False


def _fetch_index(etag: str | None) -> _Fetched | None:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    request = Request(registry_url())
    if etag:
        request.add_header("If-None-Match", etag)
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - URL comes from config
            decoded = json.loads(response.read().decode("utf-8"))
            fresh_etag = response.headers.get("ETag")
    except HTTPError as error:
        if error.code == 304:
            return _Fetched(entries=[], etag=etag, unchanged=True)
        return None
    except (URLError, OSError, ValueError):
        return None
    raw = as_object(decoded)
    entries = _entries(raw.get("entries") if raw is not None else decoded)
    return None if entries is None else _Fetched(entries=entries, etag=fresh_etag)


def _entries(value: object) -> list[JsonObject] | None:
    items = as_array(value)
    if items is None:
        return None
    return [entry for entry in (as_object(i) for i in items) if entry is not None]
