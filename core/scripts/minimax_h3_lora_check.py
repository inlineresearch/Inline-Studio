"""Render the same seed with and without a LoRA, and measure what changed.

The question a unit test cannot answer: does an adapter trained by the Trainer actually reach the
denoiser at generation time? A LoRA that changes nothing means the fuse silently missed its
targets; one that produces noise means the training convention is wrong. Both pass a test suite.

Two loads in one process, because the pipeline cache keys on the LoRA stack and evicting between
them is the same path a user takes when they wire an adapter in.

    cd core && PYTHONPATH=src .venv/bin/python scripts/minimax_h3_lora_check.py \
        --lora ../outputs/.../skin-h3.safetensors "a close-up portrait ..."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("h3lora")
OUT = Path(__file__).resolve().parents[2] / "outputs" / "minimax-h3-bench" / "lora-check"


def _render(policy: Any, prompt: str, loras: tuple[Any, ...], args: Any) -> dict[str, Any]:
    import torch

    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3.pipeline import load_pipeline, render_staged
    from inline_core.models.minimaxh3.runner import GRID
    from inline_core.models.video_params import snap_canvas, snap_frames

    width, height = snap_canvas(args.width, args.height, multiple=32, minimum=32)
    frames = snap_frames(args.seconds, GRID)

    started = time.perf_counter()
    pipe = load_pipeline(policy, params={}, partition="fl2va", loras=loras)
    load_s = round(time.perf_counter() - started, 1)

    rt.reset_peak_vram()
    started = time.perf_counter()
    state = render_staged(
        pipe, policy.placement("denoiser").device,
        prompt=prompt, num_frames=frames, height=height, width=width,
        num_inference_steps=args.steps, output_type="pil",
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    return {
        "load_s": load_s,
        "generate_s": round(time.perf_counter() - started, 1),
        "videos": state.get("videos"),
        "audio": state.get("audio"),
        "sampling_rate": state.get("sampling_rate"),
        "frames": frames,
        "size": (width, height),
    }


def _difference(a: Any, b: Any) -> dict[str, float]:
    """Mean absolute pixel difference between two clips, in 0-255 units."""
    import numpy as np

    left = np.stack([np.asarray(f, dtype=np.float32) for f in a])
    right = np.stack([np.asarray(f, dtype=np.float32) for f in b])
    delta = np.abs(left - right)
    return {
        "mean_abs": round(float(delta.mean()), 4),
        "max_abs": round(float(delta.max()), 2),
        "changed_fraction": round(float((delta > 1.0).mean()), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--lora", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--label", default="lora-check")
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from inline_core.device.memory import MemoryPolicy
    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.minimaxh3.runner import GRID
    from inline_core.runtime.video_encode import encode_video_file

    out = OUT.parent / args.label
    out.mkdir(parents=True, exist_ok=True)
    policy = MemoryPolicy()
    record: dict[str, Any] = {"prompt": args.prompt, "seed": args.seed, "steps": args.steps}

    # Base first, so the LoRA'd load is the one that has to evict a live pipeline - which is what a
    # user does when they wire an adapter into a node they have already rendered from.
    logger.info("--- rendering WITHOUT the LoRA ---")
    base = _render(policy, args.prompt, (), args)
    record["without"] = {k: base[k] for k in ("load_s", "generate_s")}
    encode_video_file(
        out / "without-lora.mp4", base["videos"][0], fps=GRID.fps,
        audio=base["audio"][0] if base["audio"] is not None and len(base["audio"]) else None,
        sample_rate=base["sampling_rate"],
    )

    logger.info("--- rendering WITH the LoRA ---")
    tuned = _render(policy, args.prompt, (LoraRef(file=args.lora, strength=args.strength),), args)
    record["with"] = {k: tuned[k] for k in ("load_s", "generate_s")}
    encode_video_file(
        out / "with-lora.mp4", tuned["videos"][0], fps=GRID.fps,
        audio=tuned["audio"][0] if tuned["audio"] is not None and len(tuned["audio"]) else None,
        sample_rate=tuned["sampling_rate"],
    )

    record["difference"] = _difference(base["videos"][0], tuned["videos"][0])
    record["lora"] = args.lora
    record["strength"] = args.strength

    # A few frames side by side, so the change can be looked at rather than only measured.
    for index in (0, base["frames"] // 2, base["frames"] - 1):
        base["videos"][0][index].save(out / f"frame{index:03d}-without.png")
        tuned["videos"][0][index].save(out / f"frame{index:03d}-with.png")

    (out / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))

    delta = record["difference"]["mean_abs"]
    if delta == 0.0:
        print("\nFAIL: identical output - the LoRA reached nothing")
        return 1
    print(f"\nclips differ by {delta} mean absolute (0-255).")
    print("Look at the frames before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
