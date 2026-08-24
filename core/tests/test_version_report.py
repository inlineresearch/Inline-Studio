"""The two halves ship separately, so the boot banner has to name both and notice either lagging."""

from __future__ import annotations

import json
from pathlib import Path

from inline_core.server import version as v


def _component(package: str, installed: str | None, origin: str = "") -> v.Component:
    return v.Component(package, installed, origin)


def test_a_newer_release_is_announced_with_a_command() -> None:
    lines = v.update_lines(_component(v.FRONTEND_PACKAGE, "1.2.52"), "1.3.0")
    assert "1.2.52 -> 1.3.0" in lines[0]
    assert v.FRONTEND_PACKAGE in lines[1]


def test_an_editable_checkout_is_told_to_pull_first() -> None:
    """Its version is recorded at install time, so pip would not move it - only the source does."""
    lines = v.update_lines(_component(v.CORE_PACKAGE, "1.2.0", "editable"), "1.3.0")
    assert "git pull" in lines[1]


def test_nothing_is_claimed_when_current_or_unreachable() -> None:
    assert v.update_lines(_component(v.CORE_PACKAGE, "1.3.0"), "1.3.0") == []
    assert v.update_lines(_component(v.CORE_PACKAGE, "1.4.0"), "1.3.0") == []
    assert v.update_lines(_component(v.CORE_PACKAGE, "1.3.0"), None) == []
    assert v.update_lines(_component(v.FRONTEND_PACKAGE, None, "local build: x"), "1.3.0") == []


def test_version_compare_without_packaging(monkeypatch) -> None:
    """`packaging` is not a declared dependency of the engine, so the fallback has to hold."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("packaging"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert v.is_newer("1.3.0", "1.2.52")
    assert v.is_newer("1.10.0", "1.9.0")
    assert not v.is_newer("1.3.0", "1.3.0")
    assert not v.is_newer("1.2.0", "1.3.0")


def test_a_local_spa_build_reports_no_package_version(tmp_path: Path) -> None:
    """A `--rebuild` / `--front-end-root` run serves source, which no wheel version describes."""
    component = v.frontend_component(str(tmp_path / "dist-web"))
    assert component.version is None
    assert "local build" in component.origin
    assert v.frontend_component(None).origin == "not installed"


def test_a_stale_cache_is_ignored(monkeypatch, tmp_path: Path) -> None:
    """Otherwise the first check's answer would be reported as the newest release forever."""
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path))
    cache = tmp_path / "version-check.json"
    cache.write_text(json.dumps({v.CORE_PACKAGE: "1.3.0"}))
    assert v._read_cache() == {v.CORE_PACKAGE: "1.3.0"}
    import os

    stale = cache.stat().st_mtime - v.CACHE_TTL_SECONDS - 1
    os.utime(cache, (stale, stale))
    assert v._read_cache() is None


def test_the_check_is_skippable(monkeypatch, capsys) -> None:
    monkeypatch.setenv("INLINE_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(v, "latest_release", lambda package: "99.0.0")
    v.report_versions(None)
    out = capsys.readouterr().out
    assert "Versions:" in out
    assert "UPDATE AVAILABLE" not in out
