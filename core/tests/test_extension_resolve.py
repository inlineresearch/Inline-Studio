"""Host-constraint generation, conflict diagnosis, and the private-site install."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from inline_core.extensions.constraints import (
    HOST_PROTECTED,
    canonical,
    conflicts,
    fingerprint,
    host_distributions,
    parse_lock,
    protected_requirements,
    prunable,
    render,
    requirement_name,
    write_constraints,
)
from inline_core.extensions.resolve import ResolutionError, resolve_and_install
from inline_core.extensions.tools import UV, missing_tools, tool_status

HOST = {"torch": "2.5.1", "numpy": "1.26.4", "einops": "0.8.0"}


# --- naming ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Huggingface_Hub", "huggingface-hub"),
        ("huggingface.hub", "huggingface-hub"),
        ("NumPy", "numpy"),
    ],
)
def test_canonical_normalizes_pep503(raw: str, expected: str) -> None:
    assert canonical(raw) == expected


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("einops>=0.7,<0.9", "einops"),
        ("torch == 2.5.1", "torch"),
        ("pillow ; python_version < '3.12'", "pillow"),
        ("mypkg @ https://example.com/mypkg.whl", "mypkg"),
        ("Torch-Vision", "torch-vision"),
    ],
)
def test_requirement_name_extracts_the_distribution(requirement: str, expected: str) -> None:
    assert requirement_name(requirement) == expected


def test_protected_requirements_flags_the_shared_ml_stack() -> None:
    """The cheap static check: an extension listing torch is blocked before any resolution runs."""
    found = protected_requirements(("einops>=0.7", "torch>=2.0", "TRANSFORMERS==4.60"))
    assert found == ["torch>=2.0", "TRANSFORMERS==4.60"]


def test_protected_set_covers_the_runtime_extra() -> None:
    for name in ("torch", "diffusers", "transformers", "numpy", "safetensors", "accelerate"):
        assert name in HOST_PROTECTED


# --- constraint file ------------------------------------------------------------------------------


def test_render_pins_every_host_package() -> None:
    text = render(HOST)
    assert "torch==2.5.1" in text
    assert "numpy==1.26.4" in text
    assert f"# host-fingerprint: {fingerprint(HOST)}" in text


def test_fingerprint_is_order_independent_and_content_sensitive() -> None:
    assert fingerprint({"a": "1", "b": "2"}) == fingerprint({"b": "2", "a": "1"})
    assert fingerprint({"a": "1"}) != fingerprint({"a": "2"})


def test_write_constraints_rewrites_only_when_the_host_changed(tmp_path: Path) -> None:
    path = tmp_path / "host-constraints.txt"
    write_constraints(path, HOST)
    first = path.stat().st_mtime_ns

    write_constraints(path, HOST)
    assert path.stat().st_mtime_ns == first, "unchanged host must not rewrite the file"

    write_constraints(path, {**HOST, "einops": "0.9.0"})
    assert path.stat().st_mtime_ns != first


def test_host_distributions_sees_the_running_interpreter() -> None:
    dists = host_distributions()
    assert "pytest" in dists


# --- conflict diagnosis ---------------------------------------------------------------------------


def test_parse_lock_reads_pins_and_ignores_comments_and_flags() -> None:
    lock = """
# via acme
--index-url https://pypi.org/simple
einops==0.8.0
    # via -r requirements.in
transformers==4.60.0 ; python_version >= '3.11'
"""
    assert parse_lock(lock) == {"einops": "0.8.0", "transformers": "4.60.0"}


def test_conflicts_names_the_package_and_both_versions() -> None:
    """The whole point: turn "resolution impossible" into something the user can act on."""
    found = conflicts("transformers==4.60.0\neinops==0.8.0\n", HOST | {"transformers": "4.52.0"})
    assert len(found) == 1
    assert found[0].name == "transformers"
    assert found[0].host_version == "4.52.0"
    assert found[0].wanted == "==4.60.0"
    assert "4.60.0" in found[0].message() and "4.52.0" in found[0].message()


def test_conflicts_ignores_packages_that_agree_with_the_host() -> None:
    assert conflicts("einops==0.8.0\n", HOST) == []


def test_conflict_marks_shared_runtime_packages_as_protected() -> None:
    found = conflicts("torch==2.7.0\n", HOST)
    assert found[0].protected is True
    assert "shared Inline runtime" in found[0].message()
    assert found[0].to_json()["protected"] is True


def test_prunable_lists_host_duplicates_and_protected_packages() -> None:
    """Pruning is what guarantees the finder can never route a shared package privately."""
    site = {"einops": "0.8.0", "numpy": "1.26.4", "humanize": "4.0.0"}
    assert prunable(site, HOST) == ["einops", "numpy"]


# --- the real install path ------------------------------------------------------------------------


def test_no_requirements_skips_resolution_entirely(tmp_path: Path) -> None:
    """Also the shape a prebuilt install reduces to: nothing private to resolve."""
    result = resolve_and_install(
        (),
        site=tmp_path / "site",
        lock_dir=tmp_path / "lock",
        constraints_path=tmp_path / "host-constraints.txt",
    )
    assert result.modules == []
    assert result.distributions == {}


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_installs_a_private_dependency_and_derives_its_modules(tmp_path: Path) -> None:
    """End-to-end against the real resolver: a pure-Python dep lands in site/ and its top-level
    module is discovered from the installed metadata."""
    result = resolve_and_install(
        ("humanize==4.9.0",),
        site=tmp_path / "site",
        lock_dir=tmp_path / "lock",
        constraints_path=tmp_path / "host-constraints.txt",
        log=tmp_path / "lock" / "install.log",
    )

    assert "humanize" in result.modules
    assert result.distributions.get("humanize") == "4.9.0"
    assert (tmp_path / "site" / "humanize").is_dir()
    assert (tmp_path / "lock" / "requirements.lock").is_file()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_requiring_a_different_host_version_fails_with_a_named_conflict(tmp_path: Path) -> None:
    """The host-override signal. numpy is installed here, so demanding an impossible version must
    fail with the package named rather than a resolver trace."""
    with pytest.raises(ResolutionError) as excinfo:
        resolve_and_install(
            ("numpy==1.11.0",),
            site=tmp_path / "site",
            lock_dir=tmp_path / "lock",
            constraints_path=tmp_path / "host-constraints.txt",
            log=tmp_path / "lock" / "install.log",
        )
    assert "numpy" in str(excinfo.value).lower()


# --- external tools -------------------------------------------------------------------------------


def test_tool_status_reports_availability_with_an_install_hint() -> None:
    statuses = {s.name: s for s in tool_status()}
    assert set(statuses) == {"git", "uv"}
    for status in statuses.values():
        assert status.hint, "a missing tool must always come with an actionable hint"
        if status.available:
            assert status.version


def test_missing_tools_matches_what_is_on_path() -> None:
    assert ("uv" in missing_tools()) is (shutil.which("uv") is None)


def test_require_raises_with_an_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError) as excinfo:
        UV.require()
    message = str(excinfo.value)
    assert "not found on PATH" in message
    assert "astral.sh/uv/install" in message, "the error must tell the user how to fix it"


# --- floating versions ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("v1.9.0", "v1.10.0"),      # numeric, not lexicographic
        ("1.2.0", "1.2.1"),
        ("v1.2.0-rc.1", "v1.2.0"),  # a prerelease never beats its release
        ("v1.2.0", "v2.0.0"),
    ],
)
def test_version_ordering(lower: str, higher: str) -> None:
    from inline_core.extensions.fetch import version_key

    low, high = version_key(lower), version_key(higher)
    assert low is not None and high is not None
    assert low < high


@pytest.mark.parametrize("tag", ["nightly", "release", "v1.2", "1.2.3.4", ""])
def test_non_release_tags_are_ignored(tag: str) -> None:
    """Floating must only ever pick a real release, never a moving branch-like tag."""
    from inline_core.extensions.fetch import version_key

    assert version_key(tag) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_latest_tag_picks_the_highest_release(tmp_path: Path) -> None:
    import subprocess

    from inline_core.extensions.fetch import latest_tag

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@t.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@t.com",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=repo, check=True, env=env
    )
    for tag in ("v1.0.0", "v1.10.0", "v1.9.0", "v2.0.0-rc.1", "nightly"):
        subprocess.run(["git", "tag", tag], cwd=repo, check=True, capture_output=True)

    assert latest_tag(f"file://{repo}") == "v1.10.0", "rc and non-release tags are skipped"


def test_latest_tag_on_an_unreachable_repo_is_none() -> None:
    from inline_core.extensions.fetch import latest_tag

    assert latest_tag("https://example.invalid/nope.git") is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_prerelease_can_still_be_installed_by_name(tmp_path: Path) -> None:
    """Floating skips prereleases, but asking for one explicitly must still work."""
    import subprocess

    from inline_core.extensions.fetch import latest_tag

    repo = tmp_path / "pre"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@t.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@t.com",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=repo, check=True, env=env
    )
    subprocess.run(["git", "tag", "v2.0.0-rc.1"], cwd=repo, check=True, capture_output=True)

    url = f"file://{repo}"
    assert latest_tag(url) is None, "a repo with only prereleases has no stable release"
    assert latest_tag(url, prereleases=True) == "v2.0.0-rc.1"
