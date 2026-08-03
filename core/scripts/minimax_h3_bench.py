"""Benchmark the local MiniMax H3 text-to-video node, one GPU at a time.

Records what the plan's runbook asks for: wall time per clip, peak VRAM, peak host RAM, and which
model files were actually loaded. Writes the clips, a settings dump and a results table into
``outputs/minimax-h3-bench/<gpu>/`` so runs on different cards can be compared side by side.

Run it with the same arguments on each card:

    cd core && .venv/bin/python scripts/minimax_h3_bench.py --label l40s
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("h3bench")

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "outputs" / "minimax-h3-bench"

#: The trained canvas, so the numbers reflect full quality rather than the fast path.
WIDTH, HEIGHT = 1344, 768
#: What ComfyUI's own H3 templates use (BasicScheduler, simple, 20).
STEPS = 20
DURATION = 10.0

CLIPS: list[dict[str, Any]] = [
    {
        "id": "01-anime",
        "seed": 101,
        "prompt": (
            "A cheerful blue robot cat with a round white face and a red collar with a golden bell "
            "pulls a glowing gadget out of the pouch on its belly, in the sunlit living room of a "
            "small Japanese house. A boy in round glasses leans in and gasps with delight. "
            "Hand-drawn 1970s anime television style, flat cel shading, thick confident ink lines, "
            "warm afternoon light through paper screens. The characters speak cheerfully in "
            "Japanese. Bright playful orchestration with a little bell chime."
        ),
    },
    {
        "id": "02-scifi",
        "seed": 202,
        "prompt": (
            "Interior of a derelict starship corridor, emergency lighting pulsing deep red through "
            "drifting dust. A lone engineer in a scuffed pressure suit walks slowly toward camera, "
            "helmet lamp cutting a hard cone through the haze, condensation on the visor. "
            "Anamorphic lens flare, shallow depth of field, 35 mm film grain, desaturated teal "
            "shadows. Distant structural groans, a low sub-bass drone, the hiss of a failing "
            "atmosphere seal."
        ),
    },
    {
        "id": "03-mountain",
        "seed": 303,
        "prompt": (
            "A vast alpine valley at first light. Camera drifts slowly forward above a ridge as low "
            "cloud pours between granite peaks and the first sun catches the snowfields in warm "
            "gold. A thin waterfall threads down the far wall. Crisp aerial cinematography, deep "
            "focus, natural colour, no people. Wind across rock, distant meltwater, a single high "
            "sustained string note."
        ),
    },
    {
        "id": "04-jp-tv",
        "seed": 404,
        "prompt": (
            "A brightly lit Japanese variety television studio. A woman in a smart blazer "
            "sits at a "
            "curved desk speaking warmly and directly to camera, gesturing to a colourful graphic "
            "panel beside her. A small studio audience laughs off camera. Glossy broadcast "
            "lighting, saturated primary colours, multi-camera television look, slight lens "
            "softness. She speaks fluent conversational Japanese with natural intonation. Light "
            "upbeat studio music underneath."
        ),
    },
    {
        "id": "05-launch",
        "seed": 505,
        "prompt": (
            "Close on the base of a heavy-lift rocket at night, held one second before ignition. "
            "Nine engines light in sequence, brilliant white-orange plumes slamming into the flame "
            "trench and boiling outward as billowing steam floods the frame. Hold-down clamps "
            "release and the vehicle begins to climb, illuminating the tower in hard moving light. "
            "Ultra detailed, high dynamic range, volumetric smoke, slow motion at the moment of "
            "ignition. Deep concussive roar building from a low rumble, crackling combustion, "
            "metallic tower resonance."
        ),
    },
]

_BED_BASE = (
    "A single continuous piece of instrumental music for a product release film. Warm analogue "
    "synth pad, soft felt piano, gentle low pulse. Restrained and modern, confident but not "
    "triumphant, no vocals, no drums, nothing abrasive. Even dynamics suitable for sitting under "
    "narration. {intent} Visually: a slow drift across a plain dark gradient backdrop, softly "
    "shifting light, no subject, no motion of any object."
)

#: H3 caps at 15 s, so a 50 s bed is four generations crossfaded together.
BED: list[dict[str, Any]] = [
    {"id": "bed-1", "seed": 901, "intent": "Slow warm opening, soft pad, distant bell tones."},
    {"id": "bed-2", "seed": 902, "intent": "Building, a low pulse enters with light ticks."},
    {"id": "bed-3", "seed": 903, "intent": "The fullest point, wide strings over the pulse."},
    {"id": "bed-4", "seed": 904, "intent": "Resolving, the pulse falls into a sustained tone."},
]
BED_SECONDS = 14.375  # 345 frames, the longest grid point
BED_CANVAS = (960, 544)  # only the audio is kept, so render small
BED_TARGET = 50.0
BED_CROSSFADE = 2.0


@dataclass
class ClipResult:
    id: str
    seed: int
    width: int
    height: int
    frames: int = 0
    seconds: float = 0.0
    load_s: float = 0.0
    generate_s: float = 0.0
    encode_s: float = 0.0
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    path: str = ""
    error: str = ""


@dataclass
class Report:
    label: str
    gpu: str = ""
    vram_total_gb: float = 0.0
    ram_total_gb: float = 0.0
    driver: str = ""
    torch: str = ""
    plan: str = ""
    models: dict[str, str] = field(default_factory=dict)
    clips: list[ClipResult] = field(default_factory=list)
    bed: list[ClipResult] = field(default_factory=list)
    total_s: float = 0.0
    bed_path: str = ""


class RamWatch:
    """Peak RSS of this process, sampled on a thread. torch only tracks the GPU side."""

    def __init__(self, interval: float = 0.5) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self.peak_gb = 0.0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> RamWatch:
        import psutil

        process = psutil.Process()

        def run() -> None:
            while not self._stop.wait(self._interval):
                self.peak_gb = max(self.peak_gb, process.memory_info().rss / 1e9)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _gpu_info() -> tuple[str, float, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        name, total, driver = (part.strip() for part in out.split(","))
        return name, float(total) / 1024, driver
    except Exception:  # noqa: BLE001 - a missing nvidia-smi must not stop the benchmark
        return "unknown", 0.0, "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="short GPU label, e.g. l40s / l4 / t4")
    parser.add_argument("--skip-bed", action="store_true", help="clips only")
    parser.add_argument("--only", default="", help="comma-separated clip ids")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import psutil
    import torch

    from inline_core.device.memory import MemoryPolicy
    from inline_core.models import pipeline_runtime as rt
    from inline_core.models.minimaxh3 import requirements as reqs
    from inline_core.models.minimaxh3.pipeline import load_pipeline
    from inline_core.models.minimaxh3.runner import GRID
    from inline_core.models.video_params import snap_canvas, snap_frames
    from inline_core.runtime.video_encode import encode_video_file

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    name, vram, driver = _gpu_info()
    report = Report(
        label=args.label, gpu=name, vram_total_gb=round(vram, 1),
        ram_total_gb=round(psutil.virtual_memory().total / 1e9, 1),
        driver=driver, torch=torch.__version__,
    )
    report.models = {
        "transformer": str(reqs.resolve_transformer("fl2va") or "missing"),
        "text_encoder": str(reqs.resolve("text_encoders", "MiniMax-H3-text-encoder") or "missing"),
        "video_vae": str(reqs.resolve("vae", reqs.VIDEO_VAE_FILE) or "missing"),
        "audio_vae": str(reqs.resolve("vae", reqs.AUDIO_VAE_FILE) or "missing"),
    }
    logger.info("GPU %s (%.1f GB), RAM %.1f GB", name, vram, report.ram_total_gb)

    # MemoryPolicy, not AutoDevicePolicy: only this one carries the fit ladder
    # (set_footprint / fit_estimate / quantization). It is what the server uses.
    policy = MemoryPolicy()
    started = time.perf_counter()

    logger.info("loading pipeline …")
    rt.reset_peak_vram()
    with RamWatch() as watch:
        load_started = time.perf_counter()
        pipe = load_pipeline(policy, params={}, partition="fl2va")
        load_s = time.perf_counter() - load_started
    fit = policy.fit_estimate()
    report.plan = fit.plan if fit else "unknown"
    logger.info("pipeline ready in %.1fs, plan=%s, peak VRAM %.1f GB, peak RAM %.1f GB",
                load_s, report.plan, rt.peak_vram_gb(), watch.peak_gb)

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    jobs = [c for c in CLIPS if not wanted or c["id"] in wanted]

    for spec in jobs:
        report.clips.append(
            _render(pipe, spec, out_dir, width=args.width, height=args.height,
                    seconds=DURATION,
                    grid=GRID, snap=snap_frames, canvas=snap_canvas, rt=rt, torch=torch,
                    encode=encode_video_file, load_s=load_s)
        )
        load_s = 0.0  # only the first clip pays the load

    if not args.skip_bed:
        for spec in BED:
            report.bed.append(
                _render(pipe, {**spec, "prompt": _BED_BASE.format(intent=spec["intent"])},
                        out_dir, width=BED_CANVAS[0], height=BED_CANVAS[1], seconds=BED_SECONDS,
                        grid=GRID, snap=snap_frames, canvas=snap_canvas, rt=rt, torch=torch,
                        encode=encode_video_file, load_s=0.0)
            )
        report.bed_path = _build_bed(out_dir, [c.path for c in report.bed if c.path])

    report.total_s = time.perf_counter() - started
    _write(report, out_dir)
    logger.info("done in %.1f min -> %s", report.total_s / 60, out_dir)
    return 0


def _render(
    pipe: Any, spec: dict[str, Any], out_dir: Path, *, width: int, height: int, seconds: float,
    grid: Any, snap: Any, canvas: Any, rt: Any, torch: Any, encode: Any, load_s: float,
) -> ClipResult:
    frames = snap(seconds, grid)
    w, h = canvas(width, height, multiple=32, minimum=32)
    result = ClipResult(
        id=spec["id"], seed=spec["seed"], width=w, height=h, frames=frames,
        seconds=round(grid.seconds(frames), 3), load_s=round(load_s, 1),
    )
    logger.info("[%s] %dx%d, %d frames (%.2fs), seed %d", spec["id"], w, h, frames,
                result.seconds, spec["seed"])
    try:
        rt.reset_peak_vram()
        with RamWatch() as watch:
            t0 = time.perf_counter()
            state = pipe(
                prompt=spec["prompt"], num_frames=frames, height=h, width=w,
                num_inference_steps=STEPS, output_type="pil",
                generator=torch.Generator(device="cpu").manual_seed(spec["seed"]),
            )
            result.generate_s = round(time.perf_counter() - t0, 1)
            result.peak_vram_gb = round(rt.peak_vram_gb(), 2)
            result.peak_ram_gb = round(watch.peak_gb, 2)

            videos, audio = state.get("videos"), state.get("audio")
            rate = state.get("sampling_rate")
            path = out_dir / f"{spec['id']}.mp4"
            t1 = time.perf_counter()
            encode(path, videos[0], fps=grid.fps,
                   audio=audio[0] if audio else None, sample_rate=rate)
            result.encode_s = round(time.perf_counter() - t1, 1)
            result.path = str(path)
        logger.info("[%s] %.1fs generate, %.1fs encode, peak VRAM %.1f GB, peak RAM %.1f GB",
                    spec["id"], result.generate_s, result.encode_s,
                    result.peak_vram_gb, result.peak_ram_gb)
    except Exception as error:  # noqa: BLE001 - one clip failing must not lose the rest
        result.error = f"{type(error).__name__}: {error}"
        logger.error("[%s] FAILED: %s", spec["id"], result.error)
        rt.free_vram()
    return result


def _build_bed(out_dir: Path, parts: list[str]) -> str:
    """Crossfade the segments' audio into one continuous track, trimmed to length."""
    from inline_core.ffmpeg import ffmpeg_exe

    exe = ffmpeg_exe()
    if exe is None or len(parts) < 2:
        return ""
    target = out_dir / "audio-bed-50s.wav"
    args: list[str] = ["-y"]
    for part in parts:
        args += ["-i", part]
    chain, previous = [], "0:a"
    for index in range(1, len(parts)):
        label = f"x{index}"
        chain.append(
            f"[{previous}][{index}:a]acrossfade=d={BED_CROSSFADE}:c1=tri:c2=tri[{label}]"
        )
        previous = label
    args += ["-filter_complex", ";".join(chain), "-map", f"[{previous}]",
             "-t", str(BED_TARGET), "-ar", "32000", "-ac", "2", str(target)]
    proc = subprocess.run([exe, *args], capture_output=True)
    if proc.returncode != 0:
        logger.error("audio bed failed: %s", proc.stderr.decode()[-400:])
        return ""
    logger.info("audio bed written: %s", target)
    return str(target)


def _write(report: Report, out_dir: Path) -> None:
    (out_dir / "results.json").write_text(json.dumps(asdict(report), indent=2))
    rows = [
        "| Clip | Canvas | Frames | Duration | Generate | Encode | Peak VRAM | Peak RAM |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for clip in report.clips + report.bed:
        if clip.error:
            rows.append(f"| {clip.id} | - | - | - | **failed** | - | - | - |")
            continue
        rows.append(
            f"| {clip.id} | {clip.width}x{clip.height} | {clip.frames} | {clip.seconds:.2f} s | "
            f"{clip.generate_s:.1f} s | {clip.encode_s:.1f} s | {clip.peak_vram_gb:.1f} GB | "
            f"{clip.peak_ram_gb:.1f} GB |"
        )
    load = report.clips[0].load_s if report.clips else 0.0
    body = f"""# MiniMax H3 benchmark: {report.label}

| | |
| --- | --- |
| GPU | {report.gpu} ({report.vram_total_gb} GB) |
| Host RAM | {report.ram_total_gb} GB |
| Driver | {report.driver} |
| torch | {report.torch} |
| Platform | {platform.platform()} |
| Memory plan chosen | `{report.plan}` |
| Pipeline load | {load:.1f} s (once, first clip only) |
| **Total wall time** | **{report.total_s / 60:.1f} min** |

## Models loaded

{chr(10).join(f"- `{k}`: `{Path(v).name}`" for k, v in report.models.items())}

## Per clip

All at 50 steps. Prompts and settings are in `../prompts.md`.

{chr(10).join(rows)}

Audio bed: `{Path(report.bed_path).name if report.bed_path else "not built"}`
"""
    (out_dir / "benchmark.md").write_text(body)


if __name__ == "__main__":
    raise SystemExit(main())
