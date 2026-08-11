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
    stubs: Path
    path: str

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
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
                **env,
            },
        )

    def uv_calls(self) -> list[str]:
        if not self.uv_log.exists():
            return []
        return [line for line in self.uv_log.read_text(encoding="utf-8").splitlines() if line]

    def make_venv_python(self) -> None:
        self.venv_python.parent.mkdir(parents=True, exist_ok=True)
        _stub(self.venv_python, 'echo "$*" >> "$PYTHON_LOG"\nexit 0\n')

    def pretend_nvidia_gpu(self, *compute_caps: str, driver: str = "") -> None:
        """A driver that answers `-L` and the compute_cap query. Pass no caps for an older driver
        that does not know the query, which is the case the fallback index exists for.

        `driver` fills the second CSV column; the launcher reads it for the R580 floor that decides
        cu130 against frozen cu128."""
        suffix = f", {driver}" if driver else ""
        answer = "".join(f"printf '{cap}{suffix}\\n'\n" for cap in compute_caps) or "exit 1\n"
        _stub(
            self.stubs / "nvidia-smi",
            f'case "$*" in\n  *compute_cap*) {answer.strip()} ;;\nesac\nexit 0\n',
        )

    def pretend_nvidia_probe_errors(self) -> None:
        """A driver that lists GPUs but answers the query with an error string. The word must never
        be coerced into a capability."""
        _stub(
            self.stubs / "nvidia-smi",
            'case "$*" in\n  *compute_cap*) printf \'Unknown Error\\n\' ;;\nesac\nexit 0\n',
        )

    def pretend_windows(self) -> None:
        """Only Windows needs an explicit CUDA index; this script runs there under Git Bash."""
        _stub(self.stubs / "uname", "printf 'MINGW64_NT-10.0-22631\\n'\nexit 0\n")

    def project_install(self) -> str:
        """The one uv call that resolves pyproject's dependencies."""
        calls = [call for call in self.uv_calls() if "-e ." in call]
        assert len(calls) == 1, self.uv_calls()
        return calls[0]


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
    # Shadows the real driver so the suite means the same thing on a GPU box as on a bare one.
    # Without it, /usr/bin/nvidia-smi leaks in and every no-GPU case silently tests the opposite.
    _stub(stubs / "nvidia-smi", "exit 1\n")
    uv_log = tmp_path / "uv.log"
    _stub(stubs / "uv", f'printf "%s\\n" "$*" >> "{uv_log}"\nexit 0\n')

    return Sandbox(
        script=core / "webui.sh",
        venv_python=core / ".venv" / "bin" / "python",
        active_env=active,
        uv_log=uv_log,
        python_log=tmp_path / "python.log",
        stubs=stubs,
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


_WHL = "--extra-index-url https://download.pytorch.org/whl"


def test_blackwell_gets_an_index_that_has_sm_120_wheels(sandbox: Sandbox) -> None:
    """The RTX 50-series bug: cu124 is frozen at torch 2.6.0 and never gained sm_120, so a Blackwell
    card has to be sent somewhere else entirely."""
    sandbox.make_venv_python()
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("12.0")

    done = sandbox.run("--install", "--extra", "runtime")

    assert done.returncode == 0, done.stderr
    install = sandbox.project_install()
    assert f"{_WHL}/cu130" in install
    # Without this the pyproject pin wins and the detected index is silently ignored. The broad
    # flag, not --no-sources-package torch: that one is too new for the uv versions people have and
    # hard-errored their install. See test_every_uv_flag_the_launcher_passes_is_real.
    assert "--no-sources" in install


def test_older_cards_get_the_index_that_still_covers_them(sandbox: Sandbox) -> None:
    """cu126 is the last index built for Maxwell..Volta, so it is both the Ampere answer and the
    fallback when the driver is too old to report a compute capability at all."""
    sandbox.make_venv_python()
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("8.6")

    assert f"{_WHL}/cu126" in _install(sandbox)

    sandbox.pretend_nvidia_gpu()  # driver that does not know the query
    assert f"{_WHL}/cu126" in _install(sandbox)


def test_the_highest_capability_across_gpus_decides(sandbox: Sandbox) -> None:
    sandbox.make_venv_python()
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("8.6", "12.0")

    assert f"{_WHL}/cu130" in _install(sandbox)


def test_torch_index_override_beats_detection(sandbox: Sandbox) -> None:
    """The reporter hand-edited webui.bat because there was no way to say this; there is now."""
    sandbox.make_venv_python()
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("12.0")

    assert f"{_WHL}/cu128" in _install(sandbox, "--torch-index", "cu128")
    assert f"{_WHL}/cu128" in _install(sandbox, INLINE_TORCH_INDEX="cu128")
    assert "https://mirror.example/whl/cu128" in _install(
        sandbox, "--torch-index", "https://mirror.example/whl/cu128"
    )

    forced_cpu = _install(sandbox, "--torch-index", "cpu")
    assert "--extra-index-url" not in forced_cpu


def test_linux_keeps_the_default_pypi_wheels(sandbox: Sandbox) -> None:
    """Linux torch on PyPI already bundles CUDA, so naming an index there would only pin us to an
    older build than the default one."""
    sandbox.make_venv_python()
    sandbox.pretend_nvidia_gpu("12.0")

    done = sandbox.run("--install", "--extra", "runtime")

    assert done.returncode == 0, done.stderr
    assert "--extra-index-url" not in sandbox.project_install()
    # Not because detection failed - the GPU was found, it just does not need an index here.
    assert "NVIDIA GPU detected" in done.stdout


def _install(sandbox: Sandbox, *args: str, **env: str) -> str:
    sandbox.uv_log.unlink(missing_ok=True)
    done = sandbox.run("--install", "--extra", "runtime", *args, **env)
    assert done.returncode == 0, done.stderr
    return sandbox.project_install()


def test_launch_prefers_our_venv_over_the_active_environment(sandbox: Sandbox) -> None:
    """The second leak: on-demand installs (torchao, the frontend) ran in the activated venv too."""
    sandbox.make_venv_python()

    done = sandbox.run("--smart-memory")

    assert done.returncode == 0, done.stderr
    ran = sandbox.python_log.read_text(encoding="utf-8")
    assert "FOREIGN" not in ran
    assert "-m inline_core.server" in ran
    assert "not the environment active in this shell" in done.stdout


# --- the detect-only flag, and the decisions it reports -----------------------------------------
#
# Two field reports were unfalsifiable because the launcher printed a conclusion and never the
# evidence. --print-torch-index prints both and installs nothing.


def _decision(sandbox: Sandbox, *args: str) -> dict[str, str]:
    result = sandbox.run("--print-torch-index", *args)
    assert result.returncode == 0, result.stderr
    out = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(": ")
        out[key] = value
    return out


def test_print_torch_index_installs_nothing(sandbox: Sandbox) -> None:
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("12.0", driver="610.88")
    _decision(sandbox)
    assert sandbox.uv_calls() == []


def test_blackwell_on_a_current_driver_gets_cu130(sandbox: Sandbox) -> None:
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("12.0", driver="610.88")
    got = _decision(sandbox)
    assert got["torch-index"] == "cu130"
    assert got["capability-major"] == "12"
    assert got["reason"] == "autodetect"


def test_blackwell_on_an_old_driver_gets_frozen_cu128(sandbox: Sandbox) -> None:
    """CUDA 13 needs R580. cu128 still serves and was the first index with sm_120, so that machine
    has exactly one workable choice and should not be made to type it back to us."""
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("12.0", driver="575.57")
    got = _decision(sandbox)
    assert got["torch-index"] == "cu128"
    assert got["reason"] == "driver-floor-cu128"


def test_ada_gets_cu126(sandbox: Sandbox) -> None:
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("8.9", driver="580.00")
    assert _decision(sandbox)["torch-index"] == "cu126"


def test_a_garbage_probe_falls_back_and_shows_its_working(sandbox: Sandbox) -> None:
    """`Unknown Error` must not become a capability. A string comparison would rank it above 10 and
    hand an unknown card cu130."""
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_probe_errors()
    got = _decision(sandbox)
    assert got["torch-index"] == "cu126"
    assert got["capability-major"] == "unknown"
    assert "Unknown Error" in got["probe"]  # the raw line is echoed, not swallowed


def test_no_gpu_reports_why(sandbox: Sandbox) -> None:
    got = _decision(sandbox)
    assert got["torch-index"] == "cpu"
    assert got["reason"] == "no-gpu"


def test_an_explicit_index_is_reported_as_an_override(sandbox: Sandbox) -> None:
    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("12.0", driver="610.88")
    got = _decision(sandbox, "--torch-index", "cu126")
    assert got["torch-index"] == "cu126"
    assert got["reason"] == "override"


# --- the flags we pass must exist in the uv we are talking to ------------------------------------


def test_every_uv_flag_the_launcher_passes_is_real(sandbox: Sandbox) -> None:
    """A stubbed uv accepts anything, so a flag that does not exist sails through every other test
    here and hard-errors on a user's machine. `--no-sources-package torch` did exactly that: valid
    in current uv, absent from the version people had, and the install died at the first command.

    Checked against the real uv's help rather than a hardcoded list, so it tracks whatever is
    installed.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; nothing to validate the launcher's flags against")
    help_text = subprocess.run(  # noqa: S603 - fixed argv
        [uv, "pip", "install", "--help"], capture_output=True, text=True, check=True, timeout=60
    ).stdout

    sandbox.pretend_windows()
    sandbox.pretend_nvidia_gpu("8.6", driver="580.82")
    sandbox.run("--install", "--extra", "runtime")

    passed = {
        word
        for call in sandbox.uv_calls()
        for word in call.split()
        if word.startswith("--")
    }
    assert passed, "the launcher ran no uv command, so this test proves nothing"
    missing = sorted(flag for flag in passed if flag not in help_text)
    assert not missing, f"webui.sh passes flags this uv does not have: {missing}"
