"""Trainer entry point: headless LoRA training without the Studio UI.

Two invocation styles share the same training loop:

1. **Manifest** (what the Trainer tab already spawns)::

       python -m inline_core.training /path/to/manifest.json

2. **Flags** (folder of stills or clips → adapter, no server)::

       python -m inline_core.training \\
         --dataset /data/clips \\
         --arch minimax-h3 \\
         --clip-seconds 1 \\
         --models-dir ./models \\
         --output ./models/loras/my_h3.safetensors \\
         --steps 500 --resolution 512

Progress is JSON lines on stdout (``protocol.py``). SIGTERM asks the loop to flush a checkpoint
and stop so the run is resumable with ``--resume`` against the same ``--work-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import protocol


def _message_for(error: Exception, manifest: dict[str, Any]) -> str:
    """Turn a CUDA OOM into something the user can act on. Training resolution drives peak VRAM far
    more than rank does, so point at that first - a 16GB card trains Z-Image at 512 but not 768."""
    if type(error).__name__ != "OutOfMemoryError":
        return str(error)
    resolution = int(manifest.get("hyperparams", {}).get("resolution", 0) or 0)
    lower = " Try 512." if resolution > 512 else ""
    at = f" at {resolution}px" if resolution else ""
    # Krea 2 is 12.9B against Z-Image's 6B, so the honest advice is different: no resolution saves a
    # card that cannot hold the base at all.
    floor = (
        " Krea 2 is a 12.9B model - training it needs roughly 40GB of VRAM."
        if manifest.get("arch") == "krea2"
        else ""
    )
    h3 = (
        " MiniMax H3 peaks around 20.6GB on the caption pass - a 24GB card is the practical floor."
        if manifest.get("arch") == "minimax-h3"
        else ""
    )
    return (
        f"Ran out of GPU memory training{at}. Lower the training resolution "
        f"(it drives peak VRAM far more than rank).{lower}{floor}{h3}"
    )


def _parse_gpu_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inline_core.training",
        description=(
            "Train a LoRA headlessly. Pass a manifest.json (Trainer-tab shape) or a --dataset "
            "folder of captioned stills/clips. MiniMax H3 learns motion when the folder holds "
            "clips and --clip-seconds is set (1s is the practical floor)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m inline_core.training run/manifest.json\n"
            "  python -m inline_core.training --dataset ./clips --arch minimax-h3 "
            "--clip-seconds 1 --models-dir ./models --output ./models/loras/style.safetensors\n"
        ),
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Existing run manifest.json (same shape the Trainer tab writes).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Folder of NNNN.png/mp4 + NNNN.txt pairs (or metadata.jsonl captions).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Models root (diffusion_models/, vae/, …). Default: INLINE_MODELS_DIR or ./models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the finished .safetensors LoRA.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for the staged dataset, checkpoints, and manifest.",
    )
    parser.add_argument(
        "--arch",
        choices=("z-image", "krea2", "flux2", "minimax-h3"),
        default="minimax-h3",
        help="Training architecture. Default: minimax-h3 (the video model).",
    )
    parser.add_argument(
        "--base-mode",
        default=None,
        help="Base checkpoint mode (H3: raw / FL2VA). Defaults per architecture.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help=(
            "H3 only: train clips as motion. 1s is the practical floor (22 frames at 24fps). "
            "Omit for stills-only intent; short clips are still snapped to the grid."
        ),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None, help="Defaults to --rank.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--base-quant",
        choices=("auto", "none", "nf4"),
        default="auto",
        help="H3 is 4-bit only; this is accepted and forced to nf4 internally.",
    )
    parser.add_argument("--offload", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--lora-scope", choices=("full", "attention"), default="full")
    parser.add_argument("--caption-dropout", type=float, default=0.05)
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Mirror every still (exact pixel flip, not latent flip).",
    )
    parser.add_argument("--trigger", default="", help="Prepended to every caption.")
    parser.add_argument(
        "--output-name",
        default="",
        help="Stem used when --output is omitted (writes models/loras/<name>.safetensors).",
    )
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated GPU indices (sets CUDA_VISIBLE_DEVICES). Multi-GPU uses accelerate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints in --work-dir (or the manifest's checkpointDir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage the dataset and write the manifest, then print its path and exit.",
    )
    return parser


def _default_models_dir() -> Path:
    env = os.environ.get("INLINE_MODELS_DIR")
    return Path(env).expanduser() if env else Path("models")


def _resolve_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Load a manifest from the positional path or build one from flags."""
    if args.manifest:
        path = Path(args.manifest)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read manifest: {exc}") from exc

    if not args.dataset:
        raise SystemExit(
            "Pass a manifest.json path or --dataset pointing at a folder of stills/clips."
        )

    from .manifest import prepare_run

    models = args.models_dir or _default_models_dir()
    # Default H3 clip length when the arch is H3 and the user did not pass the flag: 1s is the
    # practical floor and matches the Trainer Adjust panel default.
    clip_seconds = args.clip_seconds
    if args.arch == "minimax-h3" and clip_seconds is None:
        clip_seconds = 1.0

    path, manifest = prepare_run(
        dataset=args.dataset,
        models_dir=models,
        output=args.output,
        work_dir=args.work_dir,
        arch=args.arch,
        base_mode=args.base_mode,
        trigger=args.trigger,
        resume=args.resume,
        output_name=args.output_name,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_size=args.batch_size,
        resolution=args.resolution,
        save_every=args.save_every,
        base_quant=args.base_quant,
        offload=args.offload,
        lora_scope=args.lora_scope,
        caption_dropout=args.caption_dropout,
        flip_augment=args.flip,
        clip_seconds=clip_seconds,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
    )
    # Surface the path on stderr so stdout stays the pure progress protocol.
    print(f"manifest: {path}", file=sys.stderr)
    return manifest


def run_training(manifest: dict[str, Any]) -> int:
    try:
        from .trainer import train  # heavy deps (torch/diffusers/peft) load here, not at import
    except Exception as exc:  # noqa: BLE001 - report a missing training stack cleanly
        protocol.error(f"Training stack unavailable: {exc}. Install the 'training' extra.")
        return 1

    gpu_ids = [int(g) for g in (manifest.get("gpuIds") or [])]
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        output = train(manifest)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - surface any training failure as one error line
        protocol.error(_message_for(exc, manifest))
        return 1

    if output is None:
        # A cooperative stop (cancel/SIGTERM): a checkpoint was saved, no final LoRA. Non-`done`
        # exit tells the orchestrator this run is cancelled/resumable, not complete.
        return 0
    protocol.done(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Back-compat: a single non-flag argument is the manifest path the Studio orchestrator passes.
    if len(argv) == 1 and not argv[0].startswith("-"):
        try:
            manifest = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            protocol.error(f"Could not read manifest: {exc}")
            return 2
        return run_training(manifest)

    if not argv:
        protocol.error(
            "No manifest path given. Pass manifest.json or use --dataset /path (see --help)."
        )
        return 2

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else (code if isinstance(code, int) else 2)

    try:
        manifest = _resolve_manifest(args)
    except SystemExit as exc:
        message = exc.code if isinstance(exc.code, str) else str(exc)
        if message and message not in ("0", "1", "2"):
            protocol.error(message)
        return 2
    except Exception as exc:  # noqa: BLE001 - staging failures before torch loads
        protocol.error(str(exc))
        return 2

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    return run_training(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
