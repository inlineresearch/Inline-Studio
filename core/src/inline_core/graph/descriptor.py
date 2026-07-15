"""Node descriptors: the data half of a node, served at GET /v1/models. See docs/contract.md.

The behavior half (build request, run components) lives in a NodeRunner, kept out of this data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..media import MediaKind
from .schema import PortKind


class Widget(str, Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    SEED = "seed"


@dataclass(frozen=True)
class Option:
    value: str
    label: str


@dataclass(frozen=True)
class ParamField:
    key: str
    label: str
    widget: Widget
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[Option, ...] = ()
    # A dynamic catalog Core fills from what is installed (checkpoints, loras, vae, ...).
    options_from: str | None = None
    advanced: bool = False


@dataclass(frozen=True)
class Port:
    id: str
    label: str
    kind: PortKind
    required: bool = False


@dataclass(frozen=True)
class NodeDescriptor:
    type: str
    title: str
    category: str
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()
    params: tuple[ParamField, ...] = ()
    # Only generation nodes back a Frame; source/utility nodes leave this None.
    output_kind: MediaKind | None = None
    icon: str = ""
    source: str = "builtin"
    # Internal building blocks (loaders, samplers, VAE) — served for validation/execution but never
    # offered in the UI's add-node menu. Keeps generation one-click: the user sees only high-level
    # model nodes (e.g. Z-Image Turbo); loading a diffusion model / VAE / encoder happens behind it.
    hidden: bool = False

    def output(self, port_id: str) -> Port | None:
        return next((p for p in self.outputs if p.id == port_id), None)

    def input(self, port_id: str) -> Port | None:
        return next((p for p in self.inputs if p.id == port_id), None)

    def seed_keys(self) -> tuple[str, ...]:
        return tuple(p.key for p in self.params if p.widget is Widget.SEED)

    def defaults(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.params}
