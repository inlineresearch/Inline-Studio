"""Run the engine: `python -m inline_core.server`. Registers models whose deps are installed."""

from __future__ import annotations

import logging as _logging
import os as _os

# Default PyTorch to expandable CUDA segments before torch initializes its allocator (it reads this
# env at first CUDA use). Cuts the fragmentation that makes a small allocation fail with VRAM still
# free - a common low-VRAM (T4) OOM. webui.sh sets the same default; this covers a bare
# `python -m inline_core.server`. A user-set value always wins.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# XET download acceleration tracks its bytes internally, so a model download reports no per-chunk
# progress - the popup's bar would sit at 0% then snap to done. The plain HTTP path streams a real
# fraction (and still resumes). huggingface_hub reads this at import, so it must be set before then.
# A user-set value always wins.
_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import uvicorn

from ..config import data_dir, server_host, server_port
from ..device.detect import cpu_only_torch_warning, unsupported_arch_warning
from ..device.memory import MemoryPolicy
from ..extensions.loader import LoadedExtension
from ..graph.cache import InMemoryCache
from ..graph.registry import build_default_registry
from ..models.requirements import RequirementsRegistry
from ..runtime.file_store import FileTakeStore
from ..studio import config as studio_config
from ..studio.store import StudioStore
from .app import create_app
from .bootstrap import register_models
from .frontend import resolve_frontend_root
from .rpc import EventBroadcaster, RpcRouter
from .run_store import SqliteRunStore
from .version import report_versions


def _quiet_dependency_noise() -> None:
    """Two third-party import-time warnings that are noise here, not signal.

    diffusers cannot build its torchao ``torch.load`` allowlist because torchao dropped
    ``uint4_layout``; we never load torchao-serialized checkpoints, and quantization itself is
    unaffected. torch's pytree then warns about torchao registering Enum subclasses - torchao's to
    fix, not ours. Both loggers have exactly one warning site, so this hides nothing else.
    """
    for name in ("diffusers.quantizers.torchao.torchao_quantizer", "torch.utils._pytree"):
        _logging.getLogger(name).setLevel(_logging.ERROR)


def main() -> None:
    _quiet_dependency_noise()  # before register_models, which is what pulls diffusers/torchao in
    # Resolved and reported first: model registration is slow and can crash, and the two version
    # numbers are the first thing a bug report needs.
    frontend_root = resolve_frontend_root()
    report_versions(frontend_root)
    policy = MemoryPolicy()
    registry = build_default_registry()
    data = data_dir()
    takes = data / "takes"
    take_store = FileTakeStore(takes)
    run_store = SqliteRunStore(data / "runs.db")
    # Built before model registration: extensions register their own `ext:<id>:*` channels and
    # push events while they load, so both must already exist.
    rpc = RpcRouter()
    events = EventBroadcaster()
    requirements = RequirementsRegistry()
    _, extensions = register_models(
        registry, take_store, policy, requirements=requirements, rpc=rpc, events=events
    )
    if extensions:
        print(f"Extensions: {_extension_summary(extensions)}")
    # A CPU-only torch wheel on a CUDA machine is a silent ~100x slowdown, so say it loudly here
    # rather than letting the user conclude the engine is just slow.
    torch_warning = cpu_only_torch_warning() or unsupported_arch_warning()
    if torch_warning:
        print(f"WARNING: {torch_warning}")
    # The Studio app-backend: Core is the sole native backend (projects, frames, moodboard, assets,
    # generation, fal, timeline). Every InlineStudioApi channel is handled here.
    store = StudioStore(
        studio_config.data_dir(),
        studio_config.workspace_dir(),
        default_core_url=studio_config.DEFAULT_CORE_URL,
    )
    # Before any download: huggingface_hub reads the token only from the environment.
    store.apply_hf_token()
    # Reopen whatever was open before the restart, so a browser tab left open across it keeps
    # working instead of failing every call with "No project is open."
    store.restore_last_project()
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
        requirements=requirements,
    )
    host, port = server_host(), server_port()
    print(f"Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def _extension_summary(extensions: list[LoadedExtension]) -> str:
    """One line naming what loaded and what didn't, so a broken extension is visible at boot
    rather than only inside the Extensions dialog."""
    ok = [p.extension_id for p in extensions if p.error is None]
    failed = [f"{p.extension_id} ({p.error})" for p in extensions if p.error is not None]
    parts = [f"{len(ok)} loaded"] + ([f"failed: {', '.join(failed)}"] if failed else [])
    return "; ".join(parts) + (f" [{', '.join(ok)}]" if ok else "")


if __name__ == "__main__":
    main()
