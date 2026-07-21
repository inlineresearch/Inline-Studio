"""Fetching an extension repo at a pinned commit.

A bare mirror under ``.cache/git`` is kept so re-fetching a tag is cheap. The working copy is
extracted with ``git archive``, which produces no ``.git`` directory and no hooks - nothing in a
fetched repo can execute before the scanner has seen it.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from .tools import FETCH_TIMEOUT, GIT

#: https/ssh for real repos, file:// for the extension-author dev loop (install your own checkout).
#: A bare path is still rejected, so nothing can be read as a git option.
_URL_RE = re.compile(r"^(https://|git@|file:///)[A-Za-z0-9]")
_REF_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fetched:
    source: Path
    sha: str
    ref: str


def fetch(url: str, ref: str, *, mirror: Path, dest: Path) -> Fetched:
    """Clone/refresh ``url`` into ``mirror`` and extract ``ref`` into ``dest``."""
    GIT.require()
    _validate(url, ref)
    _sync_mirror(url, mirror)
    sha = _resolve(mirror, ref)
    _extract(mirror, sha, dest)
    return Fetched(source=dest, sha=sha, ref=ref)


def remote_sha(url: str, ref: str) -> str | None:
    """The commit ``ref`` currently points at upstream, without cloning. None when unreachable -
    an update check must never fail the dialog."""
    try:
        _validate(url, ref)
        # Also request the peeled ref: an exact-match pattern alone won't return the `^{}` line.
        done = _git("ls-remote", url, ref, f"{ref}^{{}}", check=False)
    except (FetchError, OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    # An annotated tag lists its own object sha first and the commit it points to on a `^{}` line.
    # Prefer the peeled commit, since the installed sha is a commit (`_resolve` uses `^{commit}`) -
    # otherwise every annotated-tag release reads as perpetually "update available".
    peeled = next((ln for ln in lines if ln.rstrip().endswith("^{}")), None)
    line = peeled or (lines[0] if lines else "")
    sha = line.split()[0] if line else ""
    return sha or None


def _validate(url: str, ref: str) -> None:
    if not _URL_RE.match(url):
        raise FetchError(f"{url!r} is not an https or ssh git URL")
    if not _REF_RE.match(ref):
        raise FetchError(f"{ref!r} is not a valid tag, branch, or commit")


def _sync_mirror(url: str, mirror: Path) -> None:
    # A valid bare mirror has HEAD at its top; a clone interrupted mid-write leaves objects/refs but
    # no HEAD, and `git clone` would then refuse the non-empty directory ("already exists").
    if (mirror / "HEAD").is_file():
        try:
            # Re-point at the URL in case a registry entry moved the repo.
            _git("remote", "set-url", "origin", url, cwd=mirror)
            _git("fetch", "--prune", "--tags", "origin", "+refs/heads/*:refs/heads/*", cwd=mirror)
            return
        except FetchError:
            pass  # A wedged mirror must not strand reinstall; drop it and re-clone below.
    shutil.rmtree(mirror, ignore_errors=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "--bare", "--quiet", url, str(mirror))


def _resolve(mirror: Path, ref: str) -> str:
    """The full commit sha for ``ref``. Pinning by sha is what makes a version reproducible even
    if the tag is later moved."""
    for candidate in (f"refs/tags/{ref}", f"refs/heads/{ref}", ref):
        args = ("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        done = _git(*args, cwd=mirror, check=False)
        sha = done.stdout.strip()
        if done.returncode == 0 and sha:
            return sha
    raise FetchError(f"{ref!r} was not found in the repository")


def _extract(mirror: Path, sha: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [GIT.require(), "archive", "--format=tar", sha],
        cwd=mirror,
        capture_output=True,
        timeout=FETCH_TIMEOUT,
        check=False,
    )
    if done.returncode != 0:
        raise FetchError(f"could not read {sha[:7]} from the repository")
    with tarfile.open(fileobj=io.BytesIO(done.stdout), mode="r|") as archive:
        for member in archive:
            if not _safe_member(member):
                raise FetchError(f"the repository contains an unsafe path: {member.name!r}")
            archive.extract(member, dest)


def _safe_member(member: tarfile.TarInfo) -> bool:
    """Belt and braces over git's own tree rules: no absolute paths, no traversal, no links."""
    if member.issym() or member.islnk() or member.isdev():
        return False
    path = Path(member.name)
    return not path.is_absolute() and ".." not in path.parts


def _git(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [GIT.require(), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=FETCH_TIMEOUT,
        check=False,
    )
    if check and done.returncode != 0:
        raise FetchError(_tail(done.stderr) or "git failed")
    return done


def _tail(text: str, lines: int = 4) -> str:
    kept = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(kept)


#: `v1.2.3`, `1.2.3`, `v1.2.3-beta.1`. Anything else is not a release tag and is ignored.
_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


def version_key(tag: str) -> tuple[int, int, int, int, str] | None:
    """Sort key for a release tag, or None when the tag isn't one.

    A prerelease sorts below the same release (`1.2.0-rc.1` < `1.2.0`), so floating to the newest
    tag never picks a release candidate over the finished version.
    """
    match = _SEMVER_RE.match(tag.strip())
    if match is None:
        return None
    major, minor, patch, pre = match.groups()
    return (int(major), int(minor), int(patch), 0 if pre else 1, pre or "")


def latest_tag(url: str, *, prereleases: bool = False) -> str | None:
    """The newest stable release tag upstream, or None when there is none (or it's unreachable).

    This is how a listing floats: the registry names the repository, and the newest tag is resolved
    here, so an author publishes by tagging rather than by opening a registry PR.

    Prereleases are skipped by default. ``v2.0.0-rc.1`` is semver-newer than ``v1.10.0``, but
    floating a user onto a release candidate they never asked for is not; install one by naming
    its tag explicitly.
    """
    try:
        _validate(url, "HEAD")
        done = _git("ls-remote", "--tags", "--refs", url, check=False)
    except (FetchError, OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None

    best: tuple[tuple[int, int, int, int, str], str] | None = None
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        tag = parts[1].removeprefix("refs/tags/")
        key = version_key(tag)
        if key is None or (not prereleases and key[3] == 0):
            continue
        if best is None or key > best[0]:
            best = (key, tag)
    return best[1] if best else None
