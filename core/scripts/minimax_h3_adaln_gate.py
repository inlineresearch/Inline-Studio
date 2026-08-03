"""Phase 9 gate: does factorising the AdaLN branch change what MiniMax H3 renders?

The per-block analysis showed the factorisation reproduces each block's modulation to ~1e-4
relative. That does not settle the question, because modulation errors could compound across 50
blocks and every denoising step. So this renders the **same seed twice** and compares the pixels.

**Both renders are full precision.** int8 on either side would fold two effects into one number and
leave the tolerance describing neither: rounding every projection to 8 bits is a much larger
perturbation than the factorisation being measured, and it is applied to different layers on the two
sides. Full bf16 both ways isolates the factorisation exactly, which is the only number worth
stating a tolerance against.

The conditioner stays 4-bit, because 66.7 GB of it in bf16 does not load anywhere. It is not part of
the comparison: both renders take the same conditioner and the same prompt, so it contributes the
same conditioning to each.

Pass criteria, fixed before the first run:

* mean absolute pixel difference below 5/255 (2%) across all frames,
* the two clips show the same scene and the same motion.

Structurally different content fails regardless of the number.

Two loads rather than one, deliberately. Swapping the modules in a loaded model would leave the
group-offload hooks holding references to the modules they replaced, so the factorised projections
would never be onloaded. Loading twice also means the factorised side runs the **production** path -
basis derived from the checkpoint, applied in the shrink callback as each block lands - rather than
a rearrangement only the gate performs.
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

logger = logging.getLogger("h3gate")

OUT = Path(__file__).resolve().parents[2] / "outputs" / "minimax-h3-bench" / "adaln-gate"

PROMPT = (
    "Breathtaking FPV drone cinematography, ultra wide lens, hyper-real natural colour, extremely "
    "fast forward flight, aggressive banking. No people. The camera screams down a granite spine "
    "just metres above the rock, rolls hard left and dives into a cloud-filled valley, then bursts "
    "out into blinding golden sunrise. Audio: roaring wind and a soaring orchestral swell."
)
SEED = 303

#: Fixed before the first run. See the module docstring.
MEAN_PIXEL_TOLERANCE = 5.0 / 255.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import numpy as np
    import torch

    from inline_core.device.memory import MemoryPolicy
    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3 import requirements as reqs
    from inline_core.models.minimaxh3.pipeline import load_pipeline
    from inline_core.models.minimaxh3.runner import GRID
    from inline_core.models.video_params import snap_canvas, snap_frames
    from inline_core.runtime.video_encode import encode_video_file

    OUT.mkdir(parents=True, exist_ok=True)
    frames = snap_frames(args.seconds, GRID)
    width, height = snap_canvas(args.width, args.height, multiple=32, minimum=32)
    policy = MemoryPolicy()

    def render(tag: str, *, factorise: bool) -> dict[str, Any]:
        rt.PIPELINES.clear()
        rt.free_vram()
        started = time.perf_counter()
        pipe = load_pipeline(
            policy, params={}, partition="fl2va", quantize=False, factorise_adaln=factorise
        )
        loaded = time.perf_counter() - started
        # tqdm redraws with a carriage return, which leaves `tail -f` looking hung for the length of
        # a render. One line per redraw, at most one a minute.
        pipe.set_progress_bar_config(file=_Newlines(sys.stderr), mininterval=60.0, ascii=True)
        weights = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters())

        rt.reset_peak_vram()
        started = time.perf_counter()
        state = pipe(
            prompt=PROMPT, num_frames=frames, height=height, width=width,
            num_inference_steps=args.steps, output_type="pil",
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        )
        elapsed = time.perf_counter() - started
        videos, audio = state.get("videos"), state.get("audio")
        path = OUT / f"{tag}.mp4"
        encode_video_file(
            path, videos[0], fps=GRID.fps,
            audio=audio[0] if audio is not None and len(audio) else None,
            sample_rate=state.get("sampling_rate"),
        )
        stack = np.stack([np.asarray(f.convert("RGB"), dtype=np.float32) for f in videos[0]])
        logger.info(
            "[%s] load %.0fs, generate %.0fs, peak VRAM %.1f GB, transformer %.1f GB -> %s",
            tag, loaded, elapsed, rt.peak_vram_gb(), weights / 1e9, path,
        )
        return {
            "seconds": elapsed, "load_seconds": loaded, "peak_vram_gb": rt.peak_vram_gb(),
            "transformer_gb": weights / 1e9, "path": str(path), "pixels": stack,
        }

    unpruned = render("unpruned", factorise=False)
    factorised = render("factorised", factorise=True)

    delta = np.abs(unpruned.pop("pixels") - factorised.pop("pixels"))
    mean_diff = float(delta.mean()) / 255.0
    passed = mean_diff < MEAN_PIXEL_TOLERANCE

    report = {
        "precision": "bf16 on both sides, no quantisation of the transformer",
        "tolerance_mean_pixel": MEAN_PIXEL_TOLERANCE,
        "mean_pixel_difference": mean_diff,
        "worst_frame_mean_difference": float(delta.mean(axis=(1, 2, 3)).max()) / 255.0,
        "max_absolute_pixel_difference": float(delta.max()) / 255.0,
        "passed": passed,
        "adaln_gb_saved": unpruned["transformer_gb"] - factorised["transformer_gb"],
        "singular_values": _spectrum(reqs.resolve_transformer("fl2va")),
        "unpruned": unpruned,
        "factorised": factorised,
        "settings": {
            "width": width, "height": height, "frames": frames,
            "steps": args.steps, "seed": SEED,
        },
    }
    (OUT / "gate.json").write_text(json.dumps(report, indent=2))
    logger.info("mean pixel difference %.5f (tolerance %.5f) -> %s",
                mean_diff, MEAN_PIXEL_TOLERANCE, "PASS" if passed else "FAIL")
    logger.info("transformer %.1f GB -> %.1f GB", unpruned["transformer_gb"],
                factorised["transformer_gb"])
    print("GATE_DONE", flush=True)
    return 0 if passed else 1


class _Newlines:
    """A stream that turns tqdm's carriage returns into newlines, so progress is greppable."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        return self._stream.write(text.replace("\r", "\n"))

    def flush(self) -> None:
        self._stream.flush()


def _spectrum(source: Path | None) -> dict[str, Any]:
    """The claim at the top of ``adaln.py``, measured: how many directions ``silu(temb)`` occupies.

    Reads only the two timestep-embedding tensors, so it costs about 60 MB of a 66 GB file.
    """
    import torch
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from safetensors import safe_open

    from inline_core.models.minimaxh3 import adaln

    if source is None:
        return {}
    with safe_open(str(source), framework="pt") as handle:
        get = handle.get_tensor
        proj_in = get("time_embedder.proj_in.weight")
        proj_out = get("time_embedder.proj_out.weight")
        embedder = TimestepEmbedding(
            in_channels=proj_in.shape[1], time_embed_dim=proj_in.shape[0], out_dim=proj_out.shape[0]
        )
        embedder.linear_1.weight.data = proj_in.float()
        embedder.linear_1.bias.data = get("time_embedder.proj_in.bias").float()
        embedder.linear_2.weight.data = proj_out.float()
        embedder.linear_2.bias.data = get("time_embedder.proj_out.bias").float()
    proj = Timesteps(num_channels=proj_in.shape[1], flip_sin_to_cos=True, downscale_freq_shift=0)
    singular, _ = adaln.decompose(proj, embedder)
    energy = (singular**2).cumsum(0) / (singular**2).sum()
    return {
        "top_16": [round(float(v), 4) for v in singular[:16]],
        "energy_in_top_8": round(float(energy[adaln.RANK - 1]), 8),
        "directions_above_1_percent": int((singular > float(singular[0]) * 0.01).sum()),
        "rank_kept": adaln.RANK,
        "torch": torch.__version__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
