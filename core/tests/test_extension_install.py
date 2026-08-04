"""The install state machine, exercised against a real git repository."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inline_core.device.memory import MemoryPolicy
from inline_core.extensions.install import Installer, InstallError, InstallRequest, Phase
from inline_core.extensions.paths import ExtensionsRoot
from inline_core.graph.registry import Registry
from inline_core.models.requirements import RequirementsRegistry
from inline_core.server.rpc import EventBroadcaster, RpcRouter

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

MANIFEST: dict[str, Any] = {
    "schema": 1,
    "id": "demo-extension",
    "name": "Demo Extension",
    "version": "1.0.0",
    "coreCompat": ">=1.0",
    "license": "MIT",
    "requirements": [],
    "entry": "inline_ext_demo_extension:register",
    "nodes": [{"type": "demo/invert"}],
}

NODE_SOURCE = '''
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import Port
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind


@inline_node(
    type="demo/invert",
    title="Invert",
    category="Image",
    inputs=(Port("image", "Image", PortKind.IMAGE, required=True),),
    outputs=(Port("image", "Image", PortKind.IMAGE),),
)
class Invert(NodeRunner):
    produces_takes = False

    def run(self, node, inputs, ctx) -> NodeResult:
        return NodeResult(outputs={"image": inputs["image"][0]})


def register(reg) -> None:
    reg.nodes(Invert)
'''


TWO_NODE_SOURCE = NODE_SOURCE.replace(
    "def register(reg) -> None:\n    reg.nodes(Invert)",
    """
@inline_node(
    type="demo/extra",
    title="Extra",
    category="Image",
    outputs=(Port("image", "Image", PortKind.IMAGE),),
)
class Extra(NodeRunner):
    produces_takes = False

    def run(self, node, inputs, ctx) -> NodeResult:
        return NodeResult(outputs={})


def register(reg) -> None:
    reg.nodes(Invert, Extra)
""",
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _renamed(manifest: dict[str, Any], extension_id: str) -> dict[str, Any]:
    """A copy of the manifest under a different extension id, which also changes its Python package.

    Tests needing a genuinely fresh import must not reuse a module name: sys.modules is global and
    process-wide, so a second install of the same extension reuses the already-imported module -
    the same constraint that makes a version switch restart-required in production.
    """
    package = "inline_ext_" + extension_id.replace("-", "_")
    return {**manifest, "id": extension_id, "entry": f"{package}:register"}


def _make_repo(tmp_path: Path, manifest: dict[str, Any], source: str = NODE_SOURCE) -> Path:
    extension_id = str(manifest["id"])
    repo = tmp_path / f"{extension_id}-repo"
    package = repo / "python" / ("inline_ext_" + extension_id.replace("-", "_"))
    package.mkdir(parents=True)
    (package / "basic.py").write_text(source, encoding="utf-8")
    (package / "__init__.py").write_text(
        "from .basic import *  # noqa: F403\n"
        "from .basic import register  # noqa: F401\n",
        encoding="utf-8",
    )
    (repo / "inline-extension.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "initial", cwd=repo)
    _git("tag", "v1.0.0", cwd=repo)
    return repo


@pytest.fixture
def installer(tmp_path: Path) -> Installer:
    from inline_core.runtime.file_store import FileTakeStore

    return Installer(
        Registry(),
        FileTakeStore(tmp_path / "takes"),
        MemoryPolicy(),
        requirements=RequirementsRegistry(),
        paths=ExtensionsRoot(tmp_path / "extensions"),
        rpc=RpcRouter(),
        events=EventBroadcaster(),
    )


def _install(installer: Installer, repo: Path, **kwargs: Any) -> Any:
    return installer._install(InstallRequest(source=f"file://{repo}", ref="v1.0.0", **kwargs))


# --- the happy path -------------------------------------------------------------------------


def test_installs_an_extension_and_registers_its_nodes(
    installer: Installer, tmp_path: Path
) -> None:
    repo = _make_repo(tmp_path, MANIFEST)

    result = _install(installer, repo)

    assert result.extension_id == "demo-extension"
    assert result.version.startswith("1.0.0+")
    assert result.node_types == ["demo/invert"]
    assert not result.needs_consent


def test_a_first_install_goes_live_without_a_restart(installer: Installer, tmp_path: Path) -> None:
    _install(installer, _make_repo(tmp_path, MANIFEST))

    descriptor = installer._registry.get("demo/invert")
    assert descriptor.title == "Invert"
    assert descriptor.source == "ext:demo-extension", "provenance is stamped by the registrar"


def test_installed_extension_is_listed_with_its_nodes(
    installer: Installer, tmp_path: Path
) -> None:
    _install(installer, _make_repo(tmp_path, MANIFEST))

    extensions = installer.list_packs()
    assert len(extensions) == 1
    assert extensions[0]["extensionId"] == "demo-extension"
    assert extensions[0]["enabled"] is True
    assert [n["type"] for n in extensions[0]["nodes"]] == ["demo/invert"]
    assert extensions[0]["nodes"][0]["enabled"] is True


def test_state_survives_a_restart(installer: Installer, tmp_path: Path) -> None:
    from inline_core.extensions.state import StateStore

    _install(installer, _make_repo(tmp_path, MANIFEST))

    reopened = StateStore(ExtensionsRoot(tmp_path / "extensions"))
    state = reopened.extension("demo-extension")
    assert state is not None
    assert state.enabled is True
    assert state.node_enabled("demo/invert", default=False) is True


def test_progress_streams_over_the_events_socket(installer: Installer, tmp_path: Path) -> None:
    queue = installer._events.add()
    _install(installer, _make_repo(tmp_path, MANIFEST))

    frames = []
    while not queue.empty():
        frames.append(queue.get_nowait())
    channels = [f["channel"] for f in frames]
    phases = [f["payload"].get("phase") for f in frames if "phase" in f["payload"]]

    assert "events:extensionInstallDone" in channels
    assert Phase.FETCH.value in phases
    assert Phase.ACTIVATE.value in phases


# --- failures leave nothing behind ----------------------------------------------------------------


def test_an_invalid_manifest_fails_without_installing(installer: Installer, tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {**MANIFEST, "coreCompat": ""})

    with pytest.raises(InstallError) as excinfo:
        _install(installer, repo)

    assert excinfo.value.phase is Phase.VALIDATE
    assert installer.list_packs() == []


def test_a_blocked_scan_aborts_and_reports_the_reason(
    installer: Installer, tmp_path: Path
) -> None:
    repo = _make_repo(tmp_path, {**MANIFEST, "requirements": ["torch>=2.0"]})

    with pytest.raises(InstallError) as excinfo:
        _install(installer, repo)

    assert excinfo.value.phase is Phase.SCAN
    assert "shared Inline runtime" in str(excinfo.value)
    assert excinfo.value.report is not None and excinfo.value.report.blocked
    assert installer.list_packs() == []


def test_staging_is_cleaned_up_after_a_failure(installer: Installer, tmp_path: Path) -> None:
    """The rollback story: nothing outside staging is touched, and staging itself is removed."""
    repo = _make_repo(tmp_path, {**MANIFEST, "requirements": ["torch>=2.0"]})

    with pytest.raises(InstallError):
        _install(installer, repo)

    staging = ExtensionsRoot(tmp_path / "extensions").staging
    assert not staging.exists() or not list(staging.iterdir())


def test_a_node_type_owned_by_core_is_refused(installer: Installer, tmp_path: Path) -> None:
    from inline_core.graph.descriptor import NodeDescriptor

    installer._registry.register(
        NodeDescriptor(type="demo/invert", title="Core Invert", category="Image")
    )

    with pytest.raises(InstallError) as excinfo:
        _install(installer, _make_repo(tmp_path, MANIFEST))

    assert excinfo.value.phase is Phase.PREFLIGHT
    assert "already provided by Inline Core" in str(excinfo.value)
    assert installer._registry.get("demo/invert").title == "Core Invert"


def test_a_raising_entry_point_fails_before_the_live_registry_is_touched(
    installer: Installer, tmp_path: Path
) -> None:
    """REGISTER runs against a scratch registry precisely so this cannot half-install."""
    broken = NODE_SOURCE + "\n\nraise RuntimeError('boom at import')\n"
    repo = _make_repo(tmp_path, _renamed(MANIFEST, "broken-extension"), source=broken)

    with pytest.raises(InstallError) as excinfo:
        _install(installer, repo)

    assert excinfo.value.phase is Phase.REGISTER
    assert excinfo.value.restart_required, "the broken module is stuck in sys.modules"
    assert not installer._registry.has("demo/invert")
    assert installer.list_packs() == []


def test_an_unknown_ref_fails_at_fetch(installer: Installer, tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, MANIFEST)

    with pytest.raises(InstallError) as excinfo:
        installer._install(InstallRequest(source=f"file://{repo}", ref="v9.9.9"))

    assert excinfo.value.phase is Phase.FETCH


def test_a_non_git_url_is_rejected(installer: Installer) -> None:
    with pytest.raises(InstallError) as excinfo:
        installer._install(InstallRequest(source="/etc/passwd", ref="main"))
    assert excinfo.value.phase is Phase.FETCH


# --- consent --------------------------------------------------------------------------------------


def test_a_consent_finding_pauses_instead_of_installing(
    installer: Installer, tmp_path: Path
) -> None:
    risky = NODE_SOURCE + "\n\nimport subprocess\n\n\ndef go():\n    subprocess.run(['ls'])\n"
    repo = _make_repo(tmp_path, MANIFEST, source=risky)

    result = _install(installer, repo)

    assert result.needs_consent
    assert result.scan is not None
    assert "subprocess" in result.scan.consent_rules()
    assert installer.list_packs() == [], "nothing is installed until the user consents"


def test_supplying_consent_completes_the_install(installer: Installer, tmp_path: Path) -> None:
    risky = NODE_SOURCE + "\n\nimport subprocess\n\n\ndef go():\n    subprocess.run(['ls'])\n"
    repo = _make_repo(tmp_path, MANIFEST, source=risky)

    result = _install(installer, repo, consents=("subprocess",))

    assert not result.needs_consent
    assert result.node_types == ["demo/invert"]


# --- lifecycle ------------------------------------------------------------------------------------


def test_disabling_an_extension_unregisters_its_nodes(installer: Installer, tmp_path: Path) -> None:
    _install(installer, _make_repo(tmp_path, MANIFEST))
    assert installer._registry.has("demo/invert")

    installer.set_enabled("demo-extension", False)

    assert not installer._registry.has("demo/invert"), "disable is hot; no restart needed"
    assert installer.list_packs()[0]["enabled"] is False


def test_a_default_off_node_is_validated_but_not_registered(
    installer: Installer, tmp_path: Path
) -> None:
    """Every declared node is imported and validated at REGISTER, but only default-on ones go
    live - and the result must report what is actually live."""
    manifest = _renamed(MANIFEST, "twonode-extension")
    manifest["nodes"] = [{"type": "demo/invert"}, {"type": "demo/extra", "defaultEnabled": False}]
    repo = _make_repo(tmp_path, manifest, source=TWO_NODE_SOURCE)

    result = _install(installer, repo)

    assert result.node_types == ["demo/invert"]
    assert not installer._registry.has("demo/extra")


def test_toggling_a_node_never_needs_a_restart(installer: Installer, tmp_path: Path) -> None:
    """The whole point of one entry point: the code is already imported, so enabling a node is
    just another register() call."""
    manifest = _renamed(MANIFEST, "toggle-extension")
    manifest["nodes"] = [{"type": "demo/invert"}, {"type": "demo/extra", "defaultEnabled": False}]
    _install(installer, _make_repo(tmp_path, manifest, source=TWO_NODE_SOURCE))

    on = installer.set_node_enabled("toggle-extension", "demo/extra", True)

    assert on["restartRequired"] is False
    assert installer._registry.has("demo/extra")

    off = installer.set_node_enabled("toggle-extension", "demo/extra", False)

    assert off["restartRequired"] is False
    assert not installer._registry.has("demo/extra")
    assert installer._registry.has("demo/invert"), "a sibling node is unaffected"


def test_uninstall_removes_the_extension_from_disk_and_state(
    installer: Installer, tmp_path: Path
) -> None:
    _install(installer, _make_repo(tmp_path, MANIFEST))

    installer.uninstall("demo-extension")

    assert installer.list_packs() == []
    assert not (tmp_path / "extensions" / "demo-extension").exists()
    assert not installer._registry.has("demo/invert")


def test_reinstall_recovers_from_a_corrupt_cached_mirror(
    installer: Installer, tmp_path: Path
) -> None:
    """A clone interrupted mid-write leaves a mirror with objects/refs but no HEAD. Uninstall never
    touches the cache, so a plain re-clone would hit "destination already exists". Reinstall must
    heal it instead of failing at fetch."""
    repo = _make_repo(tmp_path, MANIFEST)
    _install(installer, repo)
    installer.uninstall("demo-extension")

    mirror = next((installer.paths.cache / "git").glob("*.git"))
    (mirror / "HEAD").unlink()  # wedge the mirror the way an aborted clone would

    result = _install(installer, repo)

    assert result.name == "Demo Extension"
    assert (mirror / "HEAD").is_file()


def test_remote_sha_peels_an_annotated_tag_to_its_commit(tmp_path: Path) -> None:
    """An annotated tag's ls-remote lists the tag object first and the commit on a `^{}` line. The
    update check compares against the installed *commit*, so remote_sha must peel - otherwise every
    annotated-tag release reads as perpetually 'update available'."""
    from inline_core.extensions.fetch import remote_sha

    repo = tmp_path / "annotated-repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "file.txt").write_text("hi", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "initial", cwd=repo)
    _git("tag", "-a", "v1.0.0", "-m", "release", cwd=repo)  # annotated, not lightweight

    commit = subprocess.run(
        ["git", "rev-parse", "v1.0.0^{commit}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert remote_sha(f"file://{repo}", "v1.0.0") == commit


def test_switching_to_an_uninstalled_version_is_refused(
    installer: Installer, tmp_path: Path
) -> None:
    _install(installer, _make_repo(tmp_path, MANIFEST))

    with pytest.raises(InstallError, match="not installed"):
        installer.switch_version("demo-extension", "9.9.9+deadbee")


def test_rollback_repoints_current_and_requires_a_restart(
    installer: Installer, tmp_path: Path
) -> None:
    repo = _make_repo(tmp_path, MANIFEST)
    first = _install(installer, repo)

    # A second version of the same extension.
    (repo / "inline-extension.json").write_text(
        json.dumps({**MANIFEST, "version": "1.1.0"}, indent=2), encoding="utf-8"
    )
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "v1.1.0", cwd=repo)
    _git("tag", "v1.1.0", cwd=repo)
    second = installer._install(InstallRequest(source=f"file://{repo}", ref="v1.1.0"))

    assert second.version != first.version
    assert second.restart_required, "the extension was already imported this session"

    result = installer.switch_version("demo-extension", first.version)

    assert result["restartRequired"] is True
    current = ExtensionsRoot(tmp_path / "extensions").extension("demo-extension").current()
    assert current == first.version


GENERATE_SOURCE = '''
import numpy as np
from inline_core.extensions.api import inline_node
from inline_core.graph.descriptor import ParamField, Port, Widget
from inline_core.graph.runners import NodeResult, NodeRunner
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind


@inline_node(
    type="demo/invert",
    title="Generate",
    category="Generate",
    output_kind=MediaKind.IMAGE,
    outputs=(Port("image", "Image", PortKind.IMAGE),),
    params=(ParamField("size", "Size", Widget.NUMBER, 32),),
)
class Generate(NodeRunner):
    produces_takes = True

    def run(self, node, inputs, ctx) -> NodeResult:
        size = int({**Generate.__inline_descriptor__.defaults(), **node.params}["size"])
        image = np.zeros((size, size, 3), dtype=np.uint8)
        if ctx.takes is None:
            return NodeResult(outputs={"image": image})
        return NodeResult(
            outputs={"image": image}, takes=[ctx.takes.save(ctx.run_id, node.id, image, {})]
        )


def register(reg) -> None:
    reg.nodes(Generate)
'''


def test_an_extension_node_runs_as_a_graph_and_produces_a_take(
    installer: Installer, tmp_path: Path
) -> None:
    """An extension node is runnable exactly like a built-in Generate node: `output_kind` gives it
    the Run control and take history, and `ctx.takes` is how a registrar-built runner saves."""
    from inline_core.device.memory import MemoryPolicy
    from inline_core.graph.cache import InMemoryCache
    from inline_core.graph.executor import Executor
    from inline_core.graph.schema import parse_graph
    from inline_core.media import MediaKind
    from inline_core.runtime.context import CancelToken, ExecutionContext
    from inline_core.runtime.file_store import FileTakeStore
    from inline_core.runtime.progress import ProgressEmitter
    from inline_core.runtime.run import RunState

    repo = _make_repo(tmp_path, _renamed(MANIFEST, "gen-extension"), source=GENERATE_SOURCE)
    _install(installer, repo)

    descriptor = installer._registry.get("demo/invert")
    assert descriptor.output_kind is MediaKind.IMAGE, "this is what puts Run on the node"
    assert installer._registry.runner("demo/invert").produces_takes

    class _Silent(ProgressEmitter):
        def emit(self, event: Any) -> None:
            pass

    takes_dir = tmp_path / "run-takes"
    store = FileTakeStore(takes_dir)
    graph = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [{"id": "n1", "type": "demo/invert", "params": {"size": 16}}],
            "edges": [],
        }
    )
    ctx = ExecutionContext(
        run_id="run_x",
        policy=MemoryPolicy(),
        emitter=_Silent(),
        cancel=CancelToken(),
        takes=store,
    )
    Executor(installer._registry, InMemoryCache()).run(
        graph, "n1", ctx, RunState(run_id="run_x", target="n1")
    )

    written = list(takes_dir.glob("*.png"))
    assert len(written) == 1, "the run wrote exactly one take"


def test_every_channel_uses_the_installers_root_not_the_environment(
    installer: Installer, tmp_path: Path
) -> None:
    """A regression guard: handlers once resolved the root from the environment, so `versions` and
    `registryIndex` read and wrote a different directory than the installer used - which quietly
    littered the repo's default `./extensions` during tests."""
    import asyncio

    from inline_core.extensions.handlers import register_extension_handlers

    rpc = RpcRouter()
    register_extension_handlers(rpc, installer)
    _install(installer, _make_repo(tmp_path, _renamed(MANIFEST, "root-extension")))

    async def call(channel: str, args: list[Any]) -> Any:
        out = await rpc.dispatch(channel, args)
        assert out["ok"], out
        return out["value"]

    versions = asyncio.run(call("ext:manage:versions", ["root-extension"]))
    assert versions["versions"], "versions must be read from the installer's own root"

    # The registry cache must land under the installer's root, never the process default. Assert
    # against the *configured* default, not just the cwd: conftest points that default at a tmp dir,
    # so a handler reading the environment would now write there and a cwd-only check would miss it.
    from inline_core.config import extensions_dir

    asyncio.run(call("ext:manage:registryIndex", []))
    default_root = extensions_dir()
    leaked = sorted(default_root.rglob("registry.json")) if default_root.exists() else []
    assert not leaked, f"wrote the registry cache to the process default: {leaked}"
    assert not (Path.cwd() / "extensions").exists(), "wrote to the checkout's ./extensions"
    assert (installer.paths.cache / "registry.json").is_file(), "not under the installer's root"
