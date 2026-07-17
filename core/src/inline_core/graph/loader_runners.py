"""Runners for the ``load/*`` primitives — the decomposed loader subnodes on the canvas.

A Load node resolves a **chosen model file** into a typed handle (``ComponentRef``) that threads
across a ``model`` / ``vae`` / ``text-encoder`` edge into a model runner (e.g. Z-Image). It is
deliberately **deferred + torch-free**: the node just picks the file; the heavy weight load and its
dtype/placement stay with the consuming runner, where the device policy owns placement and the
loader core (``models/loaders.py``) caches by ``(arch, kind, file, dtype)``. So a Load node is safe
"point at this file" that type-checks on the canvas and reuses the exact same load path as the model
node's own dropdowns.

Only ``z-image`` is wired today, so the arch is fixed here; Flux slots in as a new arch (plus, when
more than one exists, an arch selector or inference from the wired diffusion model).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import models_dir
from ..errors import ComponentError
from .primitives import LOAD_DIFFUSION_MODEL, LOAD_TEXT_ENCODER, LOAD_VAE
from .runners import NodeResult, NodeRunner
from .schema import Node

if TYPE_CHECKING:
    from ..runtime.context import ExecutionContext
    from .registry import Registry

# Same weight extensions the catalog scan treats as models — so a Load node's auto-pick matches what
# the dropdown lists.
_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")
_ARCH = "z-image"


@dataclass(frozen=True)
class ComponentRef:
    """A resolved reference to one model file: its kind, arch, and absolute path. Emitted by a
    ``load/*`` node and consumed by a model runner, which does the actual (cached) weight load."""

    kind: str  # "diffusion" | "vae" | "text_encoder"
    arch: str
    file: str


def _resolve_file(category: str, chosen: str) -> Path:
    """The single weight file a Load node points at: the explicit dropdown pick, else the first
    weight file in ``models/<category>/`` (mirrors the model node's "auto"). Raises if none."""
    root = models_dir() / category
    name = chosen.strip()
    if name:
        picked = root / name
        if picked.is_file():
            return picked
        raise ComponentError(f"Selected file {name!r} not found under models/{category}/.")
    if root.is_dir():
        files = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
        )
        if files:
            return files[0]
    raise ComponentError(
        f"No model file found in models/{category}/. Add one there or pick it on the node."
    )


class LoadComponentRunner(NodeRunner):
    """Resolve this node's ``file`` param into a ``ComponentRef`` on the given output port."""

    produces_takes = False

    def __init__(self, *, kind: str, category: str, output_port: str) -> None:
        self._kind = kind
        self._category = category
        self._output = output_port

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        file = _resolve_file(self._category, str(node.params.get("file", "")))
        ref = ComponentRef(kind=self._kind, arch=_ARCH, file=str(file))
        return NodeResult(outputs={self._output: ref})


def register_loaders(registry: Registry) -> None:
    """Register the ``load/*`` nodes **visible** (unhidden) with their runners, so they appear in
    the add-node menu and can feed a model node's component inputs. Torch-free — always on."""
    registry.register(
        replace(LOAD_DIFFUSION_MODEL, hidden=False),
        LoadComponentRunner(kind="diffusion", category="diffusion_models", output_port="model"),
    )
    registry.register(
        replace(LOAD_VAE, hidden=False),
        LoadComponentRunner(kind="vae", category="vae", output_port="vae"),
    )
    registry.register(
        replace(LOAD_TEXT_ENCODER, hidden=False),
        LoadComponentRunner(
            kind="text_encoder", category="text_encoders", output_port="text_encoder"
        ),
    )
