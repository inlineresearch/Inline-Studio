"""The launcher must never install into an environment it does not own.

A venv activated in the user's shell used to absorb the whole install, because uv's pip interface
targets an active ``VIRTUAL_ENV`` ahead of the local ``.venv``. Every uv call is pinned now, so
these drive the real script against a stub ``uv`` and assert on the argv it receives.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

WEBUI = Path(__file__).resolve().parents[1] / "webui.sh"

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="webui.sh is the POSIX launcher; webui.bat is its twin"
)


@dataclass(frozen=True)
class Sandbox:
    """A throwaway copy of webui.sh with a stub uv, a stub python, and a foreign venv activated."""

    script: Path
    venv_python: Path
    active_env: Path
    uv_log: Path
    python_log: Path
    path: str

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv from our own code, no shell
            ["bash", str(self.script), *args],
            cwd=self.script.parent,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={
                "PATH": self.path,
                "HOME": str(self.script.parent.parent),
                "VIRTUAL_ENV": str(self.active_env),
                "PYTHON_LOG": str(self.python_log),
            },
        )

    def uv_calls(self) -> list[str]:
        if not self.uv_log.exists():
            return []
        return [line for line in self.uv_log.read_text(encoding="utf-8").splitlines() if line]

    def make_venv_python(self) -> None:
        self.venv_python.parent.mkdir(parents=True, exist_ok=True)
        _stub(self.venv_python, 'echo "$*" >> "$PYTHON_LOG"\nexit 0\n')


def _stub(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    core = tmp_path / "core"
    core.mkdir()
    shutil.copy(WEBUI, core / "webui.sh")
    (core / ".python-version").write_text("3.11\n", encoding="utf-8")

    active = tmp_path / "comfy-venv"
    (active / "bin").mkdir(parents=True)
    _stub(active / "bin" / "python", 'echo "FOREIGN $*" >> "$PYTHON_LOG"\nexit 0\n')

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    uv_log = tmp_path / "uv.log"
    _stub(stubs / "uv", f'printf "%s\\n" "$*" >> "{uv_log}"\nexit 0\n')

    return Sandbox(
        script=core / "webui.sh",
        venv_python=core / ".venv" / "bin" / "python",
        active_env=active,
        uv_log=uv_log,
        python_log=tmp_path / "python.log",
        path=f"{stubs}:/usr/bin:/bin",
    )


def test_install_never_targets_the_active_environment(sandbox: Sandbox) -> None:
    done = sandbox.run("--install", "--extra", "all")

    assert done.returncode == 0, done.stderr
    installs = [call for call in sandbox.uv_calls() if call.startswith("pip install")]
    assert installs, sandbox.uv_calls()
    for call in installs:
        assert f"--python {sandbox.venv_python}" in call
    assert str(sandbox.active_env) not in "\n".join(sandbox.uv_calls())
    assert "will NOT be modified" in done.stdout


def test_repeat_install_reuses_the_existing_environment(sandbox: Sandbox) -> None:
    """uv venv exits non-zero on an existing environment, which used to abort the whole install."""
    sandbox.make_venv_python()

    done = sandbox.run("--install", "--extra", "runtime")

    assert done.returncode == 0, done.stderr
    assert not [call for call in sandbox.uv_calls() if call.startswith("venv")]
    assert "Reusing the existing environment" in done.stdout


def test_recreate_clears_the_environment(sandbox: Sandbox) -> None:
    sandbox.make_venv_python()

    done = sandbox.run("--install", "--recreate")

    assert done.returncode == 0, done.stderr
    assert [call for call in sandbox.uv_calls() if call.startswith("venv --clear")]


def test_use_active_env_opts_into_the_activated_environment(sandbox: Sandbox) -> None:
    done = sandbox.run("--install", "--use-active-env")

    assert done.returncode == 0, done.stderr
    installs = [call for call in sandbox.uv_calls() if call.startswith("pip install")]
    assert installs
    for call in installs:
        assert f"--python {sandbox.active_env}/bin/python" in call


def test_unknown_extra_fails_with_the_valid_list(sandbox: Sandbox) -> None:
    done = sandbox.run("--install", "--extra", "bogus")

    assert done.returncode != 0
    assert "unknown extra: bogus" in done.stderr
    assert "runtime" in done.stderr
    assert not sandbox.uv_calls()


def test_launch_prefers_our_venv_over_the_active_environment(sandbox: Sandbox) -> None:
    """The second leak: on-demand installs (torchao, the frontend) ran in the activated venv too."""
    sandbox.make_venv_python()

    done = sandbox.run("--smart-memory")

    assert done.returncode == 0, done.stderr
    ran = sandbox.python_log.read_text(encoding="utf-8")
    assert "FOREIGN" not in ran
    assert "-m inline_core.server" in ran
    assert "not the environment active in this shell" in done.stdout
