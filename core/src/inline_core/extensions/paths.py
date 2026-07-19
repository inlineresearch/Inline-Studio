"""The on-disk layout of the extensions root. One place that knows where anything lives.

::

    extensions/
      state.json              which extensions are installed, active, and node-enabled
      host-constraints.txt    snapshot of the host's installed distributions
      .cache/                 git/<id>.git bare mirrors; registry.json offline index
      .staging/<uuid>/        every install builds here; renamed into place on success
      <extension-id>/         one directory per extension, beside the metadata above
        current               pointer to the active version directory
        versions/<version>+<sha>/   source/ site/ lock/ manifest.json receipt.json
        trash/<version>/      pruned versions, removed lazily at next boot

Staging-then-rename is what makes rollback free: nothing outside ``staging/`` is touched until the
install succeeds, so a failure has nothing to undo.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import extensions_dir

#: How many non-current versions to retain per extension for rollback.
KEEP_VERSIONS = 2


def version_dirname(version: str, sha: str) -> str:
    """``1.4.0+9f2c1ab`` - unique per commit, so re-installing a moved tag never reuses a stale
    directory."""
    return f"{version}+{sha[:7]}"


@dataclass(frozen=True)
class VersionPaths:
    """One installed version of one extension."""

    root: Path

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def site(self) -> Path:
        return self.root / "site"

    @property
    def lock(self) -> Path:
        return self.root / "lock"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def receipt(self) -> Path:
        return self.root / "receipt.json"

    @property
    def titles(self) -> Path:
        """node type -> title, captured at install so a disabled node still shows a readable name
        (its descriptor isn't in the registry while it's off)."""
        return self.root / "node-titles.json"

    @property
    def owned_modules(self) -> Path:
        return self.lock / "owned-modules.json"

    @property
    def requirements_in(self) -> Path:
        return self.lock / "requirements.in"

    @property
    def requirements_lock(self) -> Path:
        return self.lock / "requirements.lock"

    @property
    def install_log(self) -> Path:
        return self.lock / "install.log"

    @property
    def scan(self) -> Path:
        return self.root / "scan.json"

    @property
    def ui(self) -> Path:
        """Served at ``/ext/<extension>/web/`` for extensions that ship a frontend bundle."""
        return self.source / "ui"

    def ensure(self) -> None:
        self.source.mkdir(parents=True, exist_ok=True)
        self.site.mkdir(parents=True, exist_ok=True)
        self.lock.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExtensionPaths:
    root: Path
    extension_id: str

    @property
    def versions(self) -> Path:
        return self.root / "versions"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def trash(self) -> Path:
        return self.root / "trash"

    @property
    def current_pointer(self) -> Path:
        return self.root / "current"

    def version(self, dirname: str) -> VersionPaths:
        return VersionPaths(self.versions / dirname)

    def installed_versions(self) -> list[str]:
        """Version directory names present on disk, newest-modified first."""
        if not self.versions.is_dir():
            return []
        entries = [p for p in self.versions.iterdir() if p.is_dir()]
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.name for p in entries]

    def current(self) -> str | None:
        """The active version directory name, or None when nothing is activated. A text file, not a
        symlink - symlinks need elevation on Windows outside developer mode."""
        try:
            name = self.current_pointer.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return name if name and (self.versions / name).is_dir() else None

    def set_current(self, dirname: str) -> None:
        write_atomic(self.current_pointer, dirname + "\n")

    def new_staging(self, token: str) -> Path:
        path = self.staging / token
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prune(self, *, keep: int = KEEP_VERSIONS) -> list[str]:
        """Move all but ``current`` + the ``keep`` most recent versions into ``trash/``. Moving, not
        deleting: on Windows a file mapped by a loaded module can't be unlinked."""
        current = self.current()
        stale = [v for v in self.installed_versions() if v != current][keep:]
        for name in stale:
            self.trash.mkdir(parents=True, exist_ok=True)
            target = self.trash / name
            shutil.rmtree(target, ignore_errors=True)
            try:
                os.replace(self.versions / name, target)
            except OSError:
                continue
        return stale

    def empty_trash(self) -> None:
        shutil.rmtree(self.trash, ignore_errors=True)


@dataclass(frozen=True)
class ExtensionsRoot:
    """The extensions root."""

    root: Path

    @classmethod
    def resolve(cls, root: Path | None = None) -> ExtensionsRoot:
        return cls((root or extensions_dir()).expanduser())

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def constraints(self) -> Path:
        return self.root / "host-constraints.txt"

    @property
    def cache(self) -> Path:
        return self.root / ".cache"

    @property
    def staging(self) -> Path:
        """Global, not per-extension: the extension id is only known once the manifest is read."""
        return self.root / ".staging"

    def new_staging(self, token: str) -> Path:
        path = self.staging / token
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear_staging(self) -> None:
        shutil.rmtree(self.staging, ignore_errors=True)

    @property
    def registry_index(self) -> Path:
        return self.cache / "registry.json"

    def git_mirror(self, extension_id: str) -> Path:
        return self.cache / "git" / f"{extension_id}.git"

    def extension(self, extension_id: str) -> ExtensionPaths:
        return ExtensionPaths(self.root / extension_id, extension_id)

    def installed_extensions(self) -> list[str]:
        """Extension directories sit at the root beside state.json. Ids are `[a-z0-9-]+`, so they
        can never collide with the dot-directories or the two metadata files."""
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.cache / "git").mkdir(parents=True, exist_ok=True)


def write_atomic(path: Path, text: str) -> None:
    """Temp file + ``os.replace``: a crash mid-write must never half-write ``state.json`` and
    strand every installed extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
