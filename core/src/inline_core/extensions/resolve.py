"""Resolves and installs an extension's dependencies into its private ``site/``.

compile against the host constraints -> install the pinned lock with ``--no-deps`` -> prune host
duplicates -> derive the owned module names from the surviving RECORD files.

Every uv call passes ``--python sys.executable``; without it uv picks an interpreter and can install
wheels for the wrong ABI.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .constraints import (
    Conflict,
    canonical,
    conflicts,
    host_distributions,
    prunable,
    write_constraints,
)
from .tools import RESOLVE_TIMEOUT, UV


class ResolutionError(RuntimeError):
    """Dependency resolution failed. ``conflicts`` is populated when the cause was a host clash."""

    def __init__(self, message: str, conflicts: list[Conflict] | None = None) -> None:
        super().__init__(message)
        self.conflicts = conflicts or []


@dataclass
class Resolution:
    """What landed in an extension's private site directory."""

    lock_text: str = ""
    #: Top-level module names the finder will route to this extension's site/.
    modules: list[str] = field(default_factory=lambda: [])
    #: Canonical distribution name -> version actually installed privately.
    distributions: dict[str, str] = field(default_factory=lambda: {})
    pruned: list[str] = field(default_factory=lambda: [])

    def to_json(self) -> dict[str, object]:
        return {
            "topLevel": sorted(self.modules),
            "distributions": self.distributions,
            "pruned": self.pruned,
        }


def resolve_and_install(
    requirements: tuple[str, ...],
    *,
    site: Path,
    lock_dir: Path,
    constraints_path: Path,
    log: Path | None = None,
) -> Resolution:
    """Resolve ``requirements`` against the host and install them into ``site``."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)

    if not requirements:
        # Runs entirely on the host stack. A prebuilt install reduces to this too.
        return Resolution()

    UV.require()
    write_constraints(constraints_path)

    requirements_in = lock_dir / "requirements.in"
    requirements_in.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    lock_path = lock_dir / "requirements.lock"

    compiled = _compile(requirements_in, lock_path, constraints_path, log)
    if not compiled:
        raise _explain_failure(requirements_in, lock_dir, log)

    _install(lock_path, site, log)
    return _finalize(site, lock_path.read_text(encoding="utf-8"))


def _compile(
    requirements_in: Path, lock_path: Path, constraints_path: Path, log: Path | None
) -> bool:
    done = _run(
        [
            "uv",
            "pip",
            "compile",
            str(requirements_in),
            "--constraint",
            str(constraints_path),
            "--python",
            sys.executable,
            "--output-file",
            str(lock_path),
        ],
        log,
    )
    return done.returncode == 0


def _explain_failure(requirements_in: Path, lock_dir: Path, log: Path | None) -> ResolutionError:
    """Turn a constrained-resolution failure into a sentence a user can act on: re-resolve
    unconstrained, then diff against the host to name the package and both versions."""
    scratch = lock_dir / "unconstrained.lock"
    done = _run(
        [
            "uv",
            "pip",
            "compile",
            str(requirements_in),
            "--python",
            sys.executable,
            "--output-file",
            str(scratch),
        ],
        log,
    )
    if done.returncode != 0:
        # Not a host clash at all - the requirements are unsatisfiable on their own terms.
        return ResolutionError(_tail(done.stderr) or "dependency resolution failed")

    try:
        found = conflicts(scratch.read_text(encoding="utf-8"))
    except OSError:
        found = []
    finally:
        scratch.unlink(missing_ok=True)

    if not found:
        return ResolutionError(
            "dependency resolution failed against the installed packages; "
            "see install.log for the resolver output"
        )
    headline = "; ".join(c.message() for c in found[:3])
    if len(found) > 3:
        headline += f" (and {len(found) - 3} more)"
    return ResolutionError(headline, conflicts=found)


def _install(lock_path: Path, site: Path, log: Path | None) -> None:
    done = _run(
        [
            "uv",
            "pip",
            "install",
            "--requirement",
            str(lock_path),
            "--target",
            str(site),
            "--python",
            sys.executable,
            "--no-deps",  # the lock is complete; never re-resolve at install time
        ],
        log,
    )
    if done.returncode != 0:
        raise ResolutionError(_tail(done.stderr) or "installing dependencies failed")


def _finalize(site: Path, lock_text: str) -> Resolution:
    """Prune host duplicates, then derive the module names the finder will own. Pruning first means
    the module list can never contain a shared package."""
    installed = _site_distributions(site)
    host = host_distributions()
    pruned: list[str] = []
    for name in prunable(installed, host):
        if _remove_distribution(site, name):
            pruned.append(name)
            installed.pop(name, None)

    return Resolution(
        lock_text=lock_text,
        modules=sorted(_top_level_modules(site)),
        distributions=installed,
        pruned=pruned,
    )


def _dist_infos(site: Path) -> list[Path]:
    return sorted(site.glob("*.dist-info")) if site.is_dir() else []


def _site_distributions(site: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for info in _dist_infos(site):
        name, _, version = info.stem.rpartition("-")
        if name:
            found[canonical(name)] = version
    return found


def _remove_distribution(site: Path, canonical_name: str) -> bool:
    """Delete a distribution's files and its dist-info, using RECORD as the file list."""
    removed = False
    for info in _dist_infos(site):
        dist_name, _, _version = info.stem.rpartition("-")
        if canonical(dist_name) != canonical_name:
            continue
        for relative in _record_paths(info):
            target = site / relative
            try:
                if target.is_file() or target.is_symlink():
                    target.unlink()
            except OSError:
                continue
        _rmtree(info)
        _prune_empty_dirs(site)
        removed = True
    return removed


def _record_paths(info: Path) -> list[str]:
    try:
        record = (info / "RECORD").read_text(encoding="utf-8")
    except OSError:
        return []
    paths: list[str] = []
    for line in record.splitlines():
        entry = line.split(",", 1)[0].strip()
        # Never follow a RECORD entry out of the site directory.
        if entry and not entry.startswith(("/", "..")) and ".." not in Path(entry).parts:
            paths.append(entry)
    return paths


def _top_level_modules(site: Path) -> set[str]:
    """Top-level importable names in ``site``. ``top_level.txt`` when present, else the first path
    segment of each RECORD entry, so a wheel without that metadata still routes."""
    modules: set[str] = set()
    for info in _dist_infos(site):
        declared = info / "top_level.txt"
        if declared.is_file():
            try:
                modules.update(
                    line.strip()
                    for line in declared.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                continue
            except OSError:
                pass
        for relative in _record_paths(info):
            head = Path(relative).parts[0] if Path(relative).parts else ""
            if not head or head.endswith(".dist-info") or head == "..":
                continue
            modules.add(head[:-3] if head.endswith(".py") else head)
    return {m for m in modules if m and not m.startswith(".")}


def _prune_empty_dirs(site: Path) -> None:
    for path in sorted(site.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()  # only succeeds when empty
            except OSError:
                continue


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _run(argv: list[str], log: Path | None) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(  # noqa: S603 - fixed argv from our own code, no shell
        argv,
        capture_output=True,
        text=True,
        timeout=RESOLVE_TIMEOUT,
        check=False,
    )
    if log is not None:
        _append_log(log, argv, done)
    return done


def _append_log(log: Path, argv: list[str], done: subprocess.CompletedProcess[str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(argv)}\n")
        if done.stdout:
            handle.write(done.stdout)
        if done.stderr:
            handle.write(done.stderr)
        handle.write(f"[exit {done.returncode}]\n\n")


def _tail(text: str, lines: int = 6) -> str:
    """The last few lines of resolver output - enough to be useful in a dialog, not a wall."""
    kept = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(kept)
