from __future__ import annotations

from pathlib import Path

from inline_core.extensions.paths import ExtensionsRoot, version_dirname
from inline_core.extensions.state import StateStore


def _paths(tmp_path: Path) -> ExtensionsRoot:
    paths = ExtensionsRoot(tmp_path / "extensions")
    paths.ensure_dirs()
    return paths


def test_activate_persists_across_reload(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = StateStore(paths)
    store.activate("acme", version="1.0.0+abc1234", owns=["einops"], nodes={"acme/x": True})

    reopened = StateStore(paths)
    state = reopened.extension("acme")
    assert state is not None
    assert state.current == "1.0.0+abc1234"
    assert state.enabled is True
    assert state.node_enabled("acme/x", default=False) is True
    assert reopened.owner_of("einops") == "acme"


def test_node_enabled_falls_back_to_the_manifest_default(tmp_path: Path) -> None:
    store = StateStore(_paths(tmp_path))
    store.activate("acme", version="1.0.0+abc1234", owns=[])
    state = store.extension("acme")
    assert state is not None
    assert state.node_enabled("acme/never-recorded", default=True) is True
    assert state.node_enabled("acme/never-recorded", default=False) is False


def test_reactivating_releases_modules_it_no_longer_needs(tmp_path: Path) -> None:
    """An extension that drops a dependency must release its claim, or the stale ownership would
    block another extension from ever owning that module."""
    paths = _paths(tmp_path)
    store = StateStore(paths)
    store.activate("acme", version="1.0.0+aaaaaaa", owns=["einops", "humanize"])
    assert store.owner_of("humanize") == "acme"

    store.activate("acme", version="2.0.0+bbbbbbb", owns=["einops"])
    assert store.owner_of("einops") == "acme"
    assert store.owner_of("humanize") is None


def test_remove_drops_the_extension_and_its_module_claims(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = StateStore(paths)
    store.activate("acme", version="1.0.0+aaaaaaa", owns=["einops"])
    store.activate("other", version="1.0.0+ccccccc", owns=["humanize"])

    store.remove("acme")
    assert store.extension("acme") is None
    assert store.owner_of("einops") is None
    assert store.owner_of("humanize") == "other"


def test_corrupt_state_file_starts_empty_rather_than_failing_boot(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.state.write_text("{not json", encoding="utf-8")
    store = StateStore(paths)
    assert store.extensions() == {}


def test_unknown_schema_is_ignored_rather_than_misread(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.state.write_text('{"schema": 99, "extensions": {"acme": {"current": "x"}}}',
    encoding="utf-8")
    assert StateStore(paths).extensions() == {}


def test_current_pointer_round_trips(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    extension = paths.extension("acme")
    dirname = version_dirname("1.4.0", "9f2c1abcdef")
    assert dirname == "1.4.0+9f2c1ab"
    extension.version(dirname).ensure()
    extension.set_current(dirname)
    assert extension.current() == dirname


def test_current_returns_none_when_the_version_directory_is_gone(tmp_path: Path) -> None:
    """Intent (state.json) and content (the version dirs) are reconciled at boot: a pointer to a
    deleted version must read as absent, not as an active install."""
    paths = _paths(tmp_path)
    extension = paths.extension("acme")
    extension.set_current("1.0.0+aaaaaaa")
    assert extension.current() is None


def test_prune_keeps_current_plus_the_most_recent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    extension = paths.extension("acme")
    names = [f"{n}.0.0+aaaaaa{n}" for n in range(1, 6)]
    for index, name in enumerate(names):
        version = extension.version(name)
        version.ensure()
        # Deterministic ordering: prune sorts newest-modified first.
        import os

        os.utime(version.root, (1_000_000 + index, 1_000_000 + index))
    extension.set_current(names[0])

    pruned = extension.prune(keep=2)

    remaining = set(extension.installed_versions())
    assert names[0] in remaining, "the active version is never pruned"
    assert len(remaining) == 3, "current + 2 retained for rollback"
    assert set(pruned).isdisjoint(remaining)
