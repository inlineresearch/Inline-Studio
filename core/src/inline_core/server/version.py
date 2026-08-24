"""What is installed and whether PyPI has newer: the engine and the UI ship separately and drift."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, cast

from ..config import data_dir

CORE_PACKAGE = "inline-core"
FRONTEND_PACKAGE = "inline-studio-frontend"

#: A day: long enough that a restart loop never hammers PyPI, short enough to notice a release.
CACHE_TTL_SECONDS = 24 * 60 * 60

#: The check runs off the boot path, but a hung socket would still hold the daemon thread open.
FETCH_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class Component:
    package: str
    #: None when what runs is not an installed distribution - a local SPA build, or no UI at all.
    version: str | None
    origin: str


def core_component() -> Component:
    """An editable install records its version at install time, so the number can lag the source."""
    return Component(
        CORE_PACKAGE, _installed(CORE_PACKAGE), "editable" if _is_editable(CORE_PACKAGE) else ""
    )


def frontend_component(frontend_root: str | None) -> Component:
    if frontend_root is None:
        return Component(FRONTEND_PACKAGE, None, "not installed")
    if _is_package_static(frontend_root):
        return Component(FRONTEND_PACKAGE, _installed(FRONTEND_PACKAGE), "")
    return Component(FRONTEND_PACKAGE, None, f"local build: {frontend_root}")


def report_versions(frontend_root: str | None) -> None:
    """Name both halves at boot, then announce an update - cached, or from a background fetch."""
    core, frontend = core_component(), frontend_component(frontend_root)
    print("Versions: " + ", ".join(describe(c) for c in (core, frontend)))
    if os.environ.get("INLINE_NO_UPDATE_CHECK", "").strip().lower() in {"1", "true", "yes"}:
        return
    cached = _read_cache()
    if cached is not None:
        _announce(core, frontend, cached)
        return
    threading.Thread(target=_check, args=(core, frontend), daemon=True).start()


def describe(component: Component) -> str:
    parts = [component.package, component.version or "unknown"]
    return " ".join(parts) + (f" ({component.origin})" if component.origin else "")


def update_lines(component: Component, latest: str | None) -> list[str]:
    """Empty unless this half is behind - an unreachable PyPI must never claim either answer."""
    if component.version is None or latest is None or not is_newer(latest, component.version):
        return []
    launcher = ".\\webui.bat" if os.name == "nt" else "./webui.sh"
    hint = f"{launcher} --install"
    if component.package == CORE_PACKAGE and component.origin == "editable":
        hint = f"git pull, then {hint}"
    return [
        f"UPDATE AVAILABLE: {component.package} {component.version} -> {latest}",
        f"  Update with: {hint}  (or: pip install -U {component.package})",
    ]


def is_newer(candidate: str, current: str) -> bool:
    """PEP 440 when packaging is importable; it is not a declared dependency of the engine."""
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(candidate) > Version(current)
        except InvalidVersion:
            pass
    except ModuleNotFoundError:
        pass
    return _numeric(candidate) > _numeric(current)


def latest_release(package: str) -> str | None:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = _object(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    latest = _object(payload.get("info")).get("version")
    return latest if isinstance(latest, str) else None


def _check(core: Component, frontend: Component) -> None:
    latest = {p: latest_release(p) for p in (CORE_PACKAGE, FRONTEND_PACKAGE)}
    found = {p: v for p, v in latest.items() if v is not None}
    # Only a complete answer is cached, so one unreachable fetch does not freeze a stale pair in.
    if len(found) == len(latest):
        _write_cache(found)
    _announce(core, frontend, found)


def _announce(core: Component, frontend: Component, latest: dict[str, str]) -> None:
    lines = [
        line
        for component in (core, frontend)
        for line in update_lines(component, latest.get(component.package))
    ]
    for line in lines:
        print(line)


def _object(value: object) -> dict[str, Any]:
    """`json.loads` returns `Any`; narrowed once here so the rest of the module stays typed."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _installed(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _is_editable(package: str) -> bool:
    try:
        raw = distribution(package).read_text("direct_url.json")
    except (PackageNotFoundError, OSError):
        return False
    if not raw:
        return False
    try:
        info = _object(json.loads(raw))
    except ValueError:
        return False
    return bool(_object(info.get("dir_info")).get("editable"))


def _is_package_static(frontend_root: str) -> bool:
    """INLINE_FRONTEND_ROOT can name the package's own static dir, which is still the package."""
    try:
        import inline_studio_frontend  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return False
    pkg_file = getattr(inline_studio_frontend, "__file__", None)
    if not pkg_file:
        return False
    static = Path(pkg_file).parent / "static"
    try:
        return static.resolve() == Path(frontend_root).resolve()
    except OSError:
        return False


def _numeric(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in value.split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _cache_path() -> Path:
    return data_dir() / "version-check.json"


def _read_cache() -> dict[str, str] | None:
    """None when absent or past the TTL, so a stale file never reports an old release as newest."""
    path = _cache_path()
    try:
        if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {k: v for k, v in _object(raw).items() if isinstance(v, str)}


def _write_cache(latest: dict[str, str]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(latest), encoding="utf-8")
    except OSError:
        pass
