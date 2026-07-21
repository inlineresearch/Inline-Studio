"""Inline Studio demo extension.

One ``register(reg)`` for the whole extension. Split nodes across modules however you like and hand
them all to the registrar here - the user toggles individual nodes, not files.
"""

from __future__ import annotations

from inline_core.extensions.api import ExtensionRegistrar

from .basic import Brightness, Invert
from .gradient import Gradient
from .upscale import Upscale


def register(reg: ExtensionRegistrar) -> None:
    reg.nodes(Gradient, Invert, Brightness, Upscale)
