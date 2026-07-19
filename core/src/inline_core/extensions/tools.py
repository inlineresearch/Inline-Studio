"""``git`` and ``uv``, which the installer shells out to and which a wheel install may lack.

Both are probed before an install starts so a missing one gives an actionable message, not a stack
trace three phases in. No silent pip fallback: a lock produced by one resolver may not reproduce
under the other, and quietly switching resolver is worse than refusing to start.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass

#: Wall-clock ceilings. Resolution over a cold wheel cache is genuinely slow; a hung network call
#: must not wedge the install thread forever.
PROBE_TIMEOUT = 10
FETCH_TIMEOUT = 300
RESOLVE_TIMEOUT = 900


class ToolMissing(RuntimeError):
    """A required external tool is not on PATH. The message is shown to the user verbatim."""


@dataclass(frozen=True)
class Tool:
    name: str
    probe: tuple[str, ...]
    why: str
    install_hint: str

    def path(self) -> str | None:
        return shutil.which(self.name)

    def available(self) -> bool:
        return self.path() is not None

    def version(self) -> str | None:
        """Version string, or None when absent or unrunnable - so a tool that is on PATH but broken
        fails here rather than mid-install."""
        if not self.available():
            return None
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
                list(self.probe),
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() or done.stderr.strip() or None if done.returncode == 0 else None

    def require(self) -> str:
        location = self.path()
        if location is None:
            raise ToolMissing(
                f"{self.name} is required to {self.why}, but it was not found on PATH.\n"
                f"{self.install_hint}"
            )
        return location


def _uv_hint() -> str:
    if platform.system() == "Windows":
        return 'Install it with:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    return "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"


def _git_hint() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Install it with:  xcode-select --install"
    if system == "Windows":
        return "Install it from:  https://git-scm.com/download/win"
    return "Install it with your package manager, e.g.:  sudo apt install git"


UV = Tool(
    name="uv",
    probe=("uv", "--version"),
    why="resolve an extension's dependencies",
    install_hint=_uv_hint(),
)

GIT = Tool(
    name="git",
    probe=("git", "--version"),
    why="download an extension from its repository",
    install_hint=_git_hint(),
)

REQUIRED = (GIT, UV)


@dataclass(frozen=True)
class ToolStatus:
    name: str
    available: bool
    version: str | None
    hint: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "hint": self.hint,
        }


def tool_status() -> list[ToolStatus]:
    """What the Extensions dialog shows when installing is unavailable."""
    return [
        ToolStatus(
            name=tool.name,
            available=tool.available(),
            version=tool.version(),
            hint=tool.install_hint,
        )
        for tool in REQUIRED
    ]


def missing_tools() -> list[str]:
    return [tool.name for tool in REQUIRED if not tool.available()]


def require_tools() -> None:
    """Raise before any install work begins if a required tool is absent."""
    for tool in REQUIRED:
        tool.require()


def can_install() -> bool:
    return not missing_tools()
