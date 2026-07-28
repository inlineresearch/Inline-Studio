"""Requirements for the client-side "Control Space" node: just the suggested ControlNet model, so
the node can offer a one-click download when ``models/controlnet/`` is empty. Torch-free (pure
filesystem), so it registers even on a runtime-less install - a download only needs the models dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import models_dir
from .requirements import ModelComponent
from .zimage.requirements import controlnet_component


class ControlSpaceProvider:
    """The ControlNet a Control Space render needs downstream - offered as a suggested download."""

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return [controlnet_component()]

    def download_target(self, component: ModelComponent) -> Path:
        return models_dir() / component.category

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        return None
