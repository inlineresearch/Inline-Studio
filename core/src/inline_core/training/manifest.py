"""Build a trainer run manifest without the Studio UI.

The Trainer tab writes the same shape from ``studio/training.py::_prepare``. Headless runs
(``python -m inline_core.training --dataset …``) use this module so a folder of stills or clips
trains through the same entry point, without starting the server or the SPA.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from . import dataset as ds

_ARCHS = ("z-image", "krea2", "flux2", "minimax-h3")
_DEFAULT_BASE = {
    "z-image": "deturbo",
    "krea2": "raw",
    "flux2": "raw",
    "minimax-h3": "raw",
}


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "lora"


def default_base_mode(arch: str) -> str:
    if arch not in _DEFAULT_BASE:
        raise ValueError(f"Unknown training architecture {arch!r}. Choose one of {_ARCHS}.")
    return _DEFAULT_BASE[arch]


def load_metadata_jsonl(folder: Path) -> dict[str, str]:
    """Optional Hugging Face-style captions: one JSON object per line with a file name and text.

    Looks for ``file_name`` / ``file`` / ``path`` and ``text`` / ``caption`` / ``prompt``. Missing
    keys are skipped; sidecars always win when both exist.
    """
    path = folder / "metadata.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        name = row.get("file_name") or row.get("file") or row.get("path")
        caption = row.get("text") or row.get("caption") or row.get("prompt")
        if not isinstance(name, str) or not isinstance(caption, str):
            continue
        out[Path(name).name] = caption.strip()
    return out


def list_media(folder: Path, *, allow_video: bool) -> list[Path]:
    """Media files in a dataset folder, images always, clips only when the arch can train them."""
    suffixes = ds._IMAGE_SUFFIXES + (ds._VIDEO_SUFFIXES if allow_video else ())
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )


def stage_dataset(
    source: Path,
    dest: Path,
    *,
    trigger: str = "",
    allow_video: bool = True,
) -> int:
    """Copy media + captions into ``dest`` as ``NNNN.ext`` / ``NNNN.txt`` pairs.

    Sidecar ``.txt`` captions win. When a file has none, ``metadata.jsonl`` in the source folder is
    used if present. ``trigger`` is prepended the same way the Studio export does.
    """
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Dataset folder not found: {source}")
    media = list_media(source, allow_video=allow_video)
    if not media:
        kinds = "images or clips" if allow_video else "images"
        raise ValueError(f"No {kinds} in {source}")

    dest.mkdir(parents=True, exist_ok=True)
    captions = load_metadata_jsonl(source)
    trigger = trigger.strip()
    for index, path in enumerate(media):
        out = dest / f"{index:04d}{path.suffix.lower()}"
        shutil.copyfile(path, out)
        sidecar = path.with_suffix(".txt")
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8").strip()
        else:
            text = captions.get(path.name, "").strip()
        caption = ", ".join(part for part in (trigger, text) if part)
        out.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return len(media)


def build_hyperparams(
    *,
    arch: str,
    base_mode: str | None = None,
    rank: int = 16,
    alpha: int | None = None,
    learning_rate: float = 1e-4,
    steps: int = 500,
    batch_size: int = 1,
    resolution: int = 512,
    save_every: int = 250,
    base_quant: str = "auto",
    offload: str = "auto",
    lora_scope: str = "full",
    caption_dropout: float = 0.05,
    flip_augment: bool = False,
    clip_seconds: float | None = None,
    output_name: str = "",
    gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Hyperparams block matching what the Trainer Adjust panel posts."""
    if arch not in _ARCHS:
        raise ValueError(f"Unknown training architecture {arch!r}. Choose one of {_ARCHS}.")
    hp: dict[str, Any] = {
        "arch": arch,
        "baseMode": base_mode or default_base_mode(arch),
        "baseQuant": base_quant,
        "offload": offload,
        "loraScope": lora_scope,
        "captionDropout": float(caption_dropout),
        "flipAugment": bool(flip_augment),
        "rank": int(rank),
        "alpha": int(alpha if alpha is not None else rank),
        "learningRate": float(learning_rate),
        "batchSize": int(batch_size),
        "steps": int(steps),
        "saveEvery": int(save_every),
        "resolution": int(resolution),
        "outputName": output_name,
        "gpuIds": list(gpu_ids or []),
    }
    # H3 only: omit for stills-only intent when the caller leaves it unset. The trainer still
    # snaps an unset clip length to the 22-frame floor for any video files present.
    if arch == "minimax-h3" and clip_seconds is not None:
        hp["clipSeconds"] = float(clip_seconds)
    return hp


def build_manifest(
    *,
    dataset_dir: Path | str,
    models_dir: Path | str,
    output_path: Path | str,
    work_dir: Path | str,
    hyperparams: dict[str, Any],
    run_id: str | None = None,
    resume: bool = False,
    trigger_word: str = "",
) -> dict[str, Any]:
    """The JSON object ``python -m inline_core.training <manifest>`` consumes."""
    work = Path(work_dir)
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    rid = run_id or uuid.uuid4().hex[:12]
    return {
        "runId": rid,
        "workingDir": str(work),
        "datasetDir": str(Path(dataset_dir)),
        "checkpointDir": str(checkpoints),
        "outputPath": str(Path(output_path)),
        "resumeFrom": str(checkpoints) if resume else None,
        "modelsDir": str(Path(models_dir)),
        "arch": hyperparams.get("arch") or "z-image",
        "baseMode": hyperparams["baseMode"],
        "triggerWord": trigger_word,
        "hyperparams": hyperparams,
        "gpuIds": hyperparams.get("gpuIds") or [],
    }


def write_manifest(manifest: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def prepare_run(
    *,
    dataset: Path | str,
    models_dir: Path | str,
    output: Path | str | None = None,
    work_dir: Path | str | None = None,
    arch: str = "minimax-h3",
    base_mode: str | None = None,
    trigger: str = "",
    resume: bool = False,
    output_name: str = "",
    **hp_kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Stage the dataset if needed, write the manifest, return ``(manifest_path, manifest)``.

    Always stages into ``work_dir/dataset`` so captions can pick up a trigger word and
    ``metadata.jsonl`` without mutating the user's folder.
    """
    source = Path(dataset).expanduser().resolve()
    rid = uuid.uuid4().hex[:12]
    work = Path(work_dir).expanduser().resolve() if work_dir else Path("training_runs") / rid
    work.mkdir(parents=True, exist_ok=True)

    hp = build_hyperparams(
        arch=arch,
        base_mode=base_mode,
        output_name=output_name,
        **hp_kwargs,
    )
    staged = work / "dataset"
    if staged.exists():
        shutil.rmtree(staged)
    count = stage_dataset(
        source,
        staged,
        trigger=trigger,
        allow_video=arch == "minimax-h3",
    )
    if count == 0:
        raise ValueError(f"No training items staged from {source}")

    models = Path(models_dir).expanduser().resolve()
    if output is not None:
        out = Path(output).expanduser().resolve()
    else:
        stem = _safe(output_name or f"{arch}-{rid}")
        out = models / "loras" / f"{stem}.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        dataset_dir=staged,
        models_dir=models,
        output_path=out,
        work_dir=work,
        hyperparams=hp,
        run_id=rid,
        resume=resume,
        trigger_word=trigger,
    )
    path = write_manifest(manifest, work / "manifest.json")
    return path, manifest
