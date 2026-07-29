"""Inline Core: the generation engine behind Inline.

Takes a typed node graph and returns immutable takes. See PLAN.md for the architecture and
docs/contract.md for the Storyline API.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    #: Resolved from the installed package, so pyproject.toml stays the only place a release is
    #: bumped. An editable install records this at install time; reinstall after bumping.
    __version__ = version("inline-core")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0"
