"""Best-effort model registration. A model whose deps are absent is skipped; the engine still serves
source nodes and the rest of the API, so a torch-less install boots cleanly.
"""

from __future__ import annotations

from ..device.policy import DevicePolicy
from ..graph.registry import Registry
from ..runtime.store import TakeStore


def register_models(registry: Registry, store: TakeStore, policy: DevicePolicy) -> list[str]:
    registered: list[str] = []
    try:
        from ..models.zimage.runner import register_zimage

        register_zimage(registry, store, policy)
        registered.append("alibaba/z-image-turbo")
    except ImportError:
        pass
    return registered
