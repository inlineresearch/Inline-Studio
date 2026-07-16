"""Run the engine: `python -m inline_core.server`. Registers models whose deps are installed."""

from __future__ import annotations

import os as _os

# Default PyTorch to expandable CUDA segments before torch initializes its allocator (it reads this
# env at first CUDA use). Cuts the fragmentation that makes a small allocation fail with VRAM still
# free — a common low-VRAM (T4) OOM. webui.sh sets the same default; this covers a bare
# `python -m inline_core.server`. A user-set value always wins.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import uvicorn

from ..config import data_dir, server_host, server_port
from ..device.memory import MemoryPolicy
from ..graph.cache import InMemoryCache
from ..graph.registry import build_default_registry
from ..runtime.file_store import FileTakeStore
from ..studio import config as studio_config
from ..studio.store import StudioStore
from .app import create_app
from .bootstrap import register_models
from .frontend import resolve_frontend_root
from .rpc import EventBroadcaster, RpcRouter
from .run_store import SqliteRunStore


def main() -> None:
    policy = MemoryPolicy()
    registry = build_default_registry()
    data = data_dir()
    takes = data / "takes"
    store = FileTakeStore(takes)
    run_store = SqliteRunStore(data / "runs.db")
    registered = register_models(registry, store, policy)
    print(f"Registered models: {registered or 'none (source nodes only)'}")
    frontend_root = resolve_frontend_root()
    fe = frontend_root or "none (API only); use --front-end-root or install the frontend package"
    print(f"Frontend: {fe}")
    # The Studio app-backend: Core is the sole native backend (projects, frames, moodboard, assets,
    # generation, fal, timeline). Every InlineStudioApi channel is handled here.
    rpc = RpcRouter()
    events = EventBroadcaster()
    store = StudioStore(
        studio_config.data_dir(),
        studio_config.workspace_dir(),
        default_comfy_url=studio_config.DEFAULT_COMFY_URL,
        default_core_url=studio_config.DEFAULT_CORE_URL,
    )
    print(f"Studio data: {studio_config.data_dir()}  |  workspace: {studio_config.workspace_dir()}")
    app = create_app(
        registry=registry,
        cache=InMemoryCache(),
        policy=policy,
        run_store=run_store,
        takes_dir=str(takes),
        frontend_root=frontend_root,
        rpc=rpc,
        events=events,
        studio_store=store,
    )
    host, port = server_host(), server_port()
    print(f"Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
