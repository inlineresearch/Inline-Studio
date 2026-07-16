"""Inline Studio — single entrypoint.

Run one Python process that serves both the API and the UI on one port:

    python main.py                                  # end users: frontend from the installed package
    python main.py --front-end-root ../Inline-Studio/dist-web   # dev: serve a local SPA build
    python main.py --listen --port 8000             # bind 0.0.0.0 on a custom port

Flags map onto the engine's INLINE_* env knobs, then hand off to the server entrypoint. Build the
dev frontend with `npm run build:spa` in the Inline Studio repo, or use `npm run dev:web` for HMR.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inline-studio",
        description="Run Inline Core + the Inline Studio UI on one port.",
    )
    # Networking
    parser.add_argument("--host", help="Bind address (default 127.0.0.1).")
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Bind 0.0.0.0 so other machines on your network can reach it.",
    )
    parser.add_argument("--port", type=int, help="Port to serve on (default 8848).")
    # Paths
    parser.add_argument(
        "--front-end-root",
        help="Serve a local frontend build (a dir with index.html) instead of the installed "
        "inline_studio_frontend package. The dev loop — rebuild locally, no republish.",
    )
    parser.add_argument("--models-dir", help="Models root to scan (default ./models).")
    parser.add_argument("--data-dir", help="Working data dir: runs + takes (default ./.inline).")
    # Device / memory
    parser.add_argument(
        "--profile", help="Device/memory profile: gpu-max | lowvram | cpu (default: auto-detected)."
    )
    parser.add_argument(
        "--lowvram", action="store_true", help="Shortcut for --profile lowvram (tight-VRAM saving)."
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Shortcut for --profile cpu (force CPU generation)."
    )
    parser.add_argument(
        "--vram-budget", type=float, help="Treat the GPU as having this many GB of usable VRAM."
    )
    # Multi-GPU (xDiT): split one image's denoise across GPUs. Auto-detected with 2+ GPUs.
    parser.add_argument(
        "--multi-gpu",
        "--parallel",
        dest="parallel",
        help="Multi-GPU split spec, e.g. pipefusion=2 or pipefusion=2,ulysses=2 (degrees multiply "
        "to the GPU count). Auto-detected with 2+ GPUs.",
    )
    args = parser.parse_args()

    if args.listen:
        os.environ["INLINE_HOST"] = "0.0.0.0"
    if args.host:
        os.environ["INLINE_HOST"] = args.host
    if args.port:
        os.environ["INLINE_PORT"] = str(args.port)
    if args.front_end_root:
        os.environ["INLINE_FRONTEND_ROOT"] = args.front_end_root
    if args.models_dir:
        os.environ["INLINE_MODELS_DIR"] = args.models_dir
    if args.data_dir:
        os.environ["INLINE_DATA_DIR"] = args.data_dir
    if args.lowvram:
        os.environ["INLINE_PROFILE"] = "lowvram"
    if args.cpu:
        os.environ["INLINE_PROFILE"] = "cpu"
    if args.profile:
        os.environ["INLINE_PROFILE"] = args.profile
    if args.vram_budget:
        os.environ["INLINE_VRAM_BUDGET_GB"] = str(args.vram_budget)
    if args.parallel:
        os.environ["INLINE_PARALLEL"] = args.parallel

    # Import after env is set so config picks it up.
    from inline_core.server.__main__ import main as run_server

    run_server()


if __name__ == "__main__":
    main()
