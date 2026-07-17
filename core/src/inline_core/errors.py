"""Typed engine errors. Fail loudly with a clear message; never catch and hide."""

from __future__ import annotations


class InlineCoreError(Exception):
    """Base for all engine errors."""


class GraphValidationError(InlineCoreError):
    """A graph failed validation before execution. Carries the offending node and port."""

    def __init__(
        self, message: str, *, node_id: str | None = None, port: str | None = None
    ) -> None:
        super().__init__(message)
        self.node_id = node_id
        self.port = port


class UnknownNodeType(GraphValidationError):
    """A node references a type absent from the registry."""


class PortTypeError(GraphValidationError):
    """An edge connects incompatible port kinds, or a required port is unwired."""


class CycleError(GraphValidationError):
    """The graph is not acyclic."""


class DeviceError(InlineCoreError):
    """A device or placement could not be satisfied."""


class ComponentError(InlineCoreError):
    """A component failed to load or execute."""


class CancelledError(InlineCoreError):
    """A run was cancelled cooperatively via its cancel token."""
