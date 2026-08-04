"""What MiniMax H3 needs on disk, and how a candidate file is recognised.

Recognition is by **safetensors header**, never by filename: `diffusion_models/` is shared across
architectures, a file can be renamed, and the published builds differ from each other in ways a name
does not carry. Reading a header is a seek and a few hundred KB, and the result is cached against
``(path, mtime, size)`` so a full models folder on a network drive does not stutter the catalog.

Torch-free and pure filesystem, like every other requirements provider: this runs on every model
popup, including on an install with no ML stack.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ...config import models_dir
from ..requirements import ModelComponent

#: Both partitions are published by Comfy-Org as one consolidated file each.
COMFY_REPO = "Comfy-Org/MiniMax-H3"
#: The tokenizer and processor are only in the original repository.
MINIMAX_REPO = "MiniMaxAI/MiniMax-H3"

FL2VA_FILE = "minimax_h3_fl2va_bf16.safetensors"
REF2VA_FILE = "minimax_h3_ref2va_bf16.safetensors"
TEXT_ENCODER_DIR = "FL2VA/text_encoder"
VIDEO_VAE_FILE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_FILE = "minimax_h3_audio_vae_fp32.safetensors"

#: The tensor every H3 transformer has, at the shape only H3 has: 3 x 56 heads x 128 into 5376.
_PROBE = "blocks.0.attn.qkv_proj.weight"
_PROBE_SHAPE = [21504, 5376]
#: Only in the pruned builds, whose AdaLN branch is a rank-8 lookup, not a projection.
_PRUNED_MARKER = "adaln_t_table"
#: ComfyUI's own quantisation, which carries scale tensors nothing else can read.
_COMFY_QUANT_SUFFIX = ".comfy_quant"


@dataclass(frozen=True)
class Candidate:
    """What a file in ``diffusion_models/`` turned out to be."""

    path: Path
    is_h3: bool
    pruned: bool = False
    comfy_quantised: bool = False

    @property
    def usable(self) -> bool:
        return self.is_h3 and not self.pruned and not self.comfy_quantised

    @property
    def reason(self) -> str:
        """Why an H3 file cannot be loaded, for the picker to show instead of hiding it."""
        if not self.is_h3:
            return "not a MiniMax H3 transformer"
        if self.comfy_quantised:
            return (
                "a ComfyUI int8 build: it carries comfy_quant scale tensors that only ComfyUI "
                "reads. Use the bf16 file; memory saving is the device policy's job here."
            )
        if self.pruned:
            return (
                "a pruned build: its AdaLN branch is a rank-8 lookup table rather than a "
                "projection, which is a different graph from the one this node runs."
            )
        return ""


def read_header(path: Path) -> dict[str, object] | None:
    """A safetensors header, or None when the file is not one."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return None
            size = struct.unpack("<Q", raw)[0]
            if not 0 < size < 200_000_000:  # a sane header; anything else is not safetensors
                return None
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    header.pop("__metadata__", None)
    return header


def inspect_file(path: Path) -> Candidate:
    """Classify a checkpoint from its header alone."""
    return _inspect_cached(str(path), *_stamp(path))


def _stamp(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (int(stat.st_mtime), stat.st_size)


@lru_cache(maxsize=512)
def _inspect_cached(path_str: str, mtime: int, size: int) -> Candidate:
    """Keyed on ``(path, mtime, size)`` so a rescan is free and a rewritten file is re-read."""
    path = Path(path_str)
    header = read_header(path)
    if header is None:
        return Candidate(path, is_h3=False)
    probe = header.get(_PROBE)
    is_h3 = isinstance(probe, dict) and list(probe.get("shape", [])) == _PROBE_SHAPE
    if not is_h3:
        return Candidate(path, is_h3=False)
    return Candidate(
        path,
        is_h3=True,
        pruned=any(_PRUNED_MARKER in key for key in header),
        comfy_quantised=any(key.endswith(_COMFY_QUANT_SUFFIX) for key in header),
    )


def usable_transformers() -> list[Path]:
    """Every H3 transformer in ``diffusion_models/`` this node can actually load."""
    root = models_dir() / "diffusion_models"
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_file() and entry.suffix == ".safetensors" and inspect_file(entry).usable
    )


def rejected_transformers() -> list[Candidate]:
    """H3 files that are present but cannot be loaded, so the picker can say why."""
    root = models_dir() / "diffusion_models"
    if not root.is_dir():
        return []
    found = [inspect_file(e) for e in sorted(root.iterdir()) if e.is_file()]
    return [c for c in found if c.is_h3 and not c.usable]


def resolve(category: str, filename: str) -> Path | None:
    path = models_dir() / category / filename
    return path if path.exists() else None


def resolve_transformer(partition: str) -> Path | None:
    """The file for a partition. Provenance is a manifest keyed by content, because the two
    partitions are structurally identical and cannot be told apart by inspection."""
    wanted = FL2VA_FILE if partition == "fl2va" else REF2VA_FILE
    direct = resolve("diffusion_models", wanted)
    if direct is not None:
        return direct
    recorded = _provenance().get(partition)
    if recorded:
        candidate = models_dir() / "diffusion_models" / recorded
        if candidate.exists():
            return candidate
    return None


def _provenance() -> dict[str, str]:
    """``partition -> filename``, written when the downloader fetched it.

    FL2VA and Ref2VA have identical keys and shapes, so a renamed file cannot be identified by
    inspection. This is the only record of which is which, and a hand-renamed file that is not in it
    falls through to the node's error rather than being guessed at.
    """
    path = models_dir() / "diffusion_models" / ".minimax-h3.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def record_provenance(partition: str, filename: str) -> None:
    """Remember which file is which partition, at download time."""
    path = models_dir() / "diffusion_models" / ".minimax-h3.json"
    current = _provenance()
    current[partition] = filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2))


def components(partition: str = "fl2va") -> list[ModelComponent]:
    """What this node needs, with live presence. Sizes in the labels because the totals are large
    enough that a user deserves to know before pressing Download."""
    ref2va_required = partition == "ref2va"
    entries: list[ModelComponent] = [
        _file("h3-fl2va", "FL2VA transformer (66.3 GB)", "diffusion_models", FL2VA_FILE,
              COMFY_REPO, f"diffusion_models/{FL2VA_FILE}", optional=ref2va_required),
        _folder("h3-text-encoder", "Text encoder, Qwen3-VL-32B (66.7 GB)", "text_encoders",
                "MiniMax-H3-text-encoder", MINIMAX_REPO, TEXT_ENCODER_DIR),
        _file("h3-video-vae", "Video VAE (5.2 GB)", "vae", VIDEO_VAE_FILE,
              COMFY_REPO, f"vae/{VIDEO_VAE_FILE}"),
        _file("h3-audio-vae", "Audio VAE (0.6 GB)", "vae", AUDIO_VAE_FILE,
              COMFY_REPO, f"vae/{AUDIO_VAE_FILE}"),
        _folder("h3-processor", "Tokenizer and processor (12 MB)", "text_encoders",
                "MiniMax-H3-processor", MINIMAX_REPO, "FL2VA/processor"),
        _file("h3-ref2va", "Ref2VA transformer (66.3 GB)", "diffusion_models", REF2VA_FILE,
              COMFY_REPO, f"diffusion_models/{REF2VA_FILE}", optional=not ref2va_required),
    ]
    return entries


def _file(
    component_id: str, label: str, category: str, filename: str,
    repo: str, repo_file: str, *, optional: bool = False,
) -> ModelComponent:
    return ModelComponent(
        id=component_id, label=label, category=category, filename=filename,
        present=(models_dir() / category / filename).is_file(),
        repo=repo, repo_file=repo_file, optional=optional,
    )


def _folder(
    component_id: str, label: str, category: str, folder: str, repo: str, repo_folder: str
) -> ModelComponent:
    return ModelComponent(
        id=component_id, label=label, category=category, filename=folder,
        present=(models_dir() / category / folder).is_dir(),
        repo=repo, repo_file="", repo_folder=repo_folder,
    )


#: The AdaLN branch is 40 percent of the checkpoint and is factorised away at load, so the file on
#: disk is not the model the policy has to place. Reporting the raw size makes the fit ladder refuse
#: machines that would have run: a 16 GB card was told it needed 72 GB when the real figure is well
#: under half that.
ADALN_SHARE = 0.392


def footprint_bytes(partition: str = "fl2va", *, factorised: bool = True) -> dict[str, int]:
    """Sizes for the fit estimate: what will actually be placed, not what is on disk."""

    def size(path: Path | None) -> int:
        try:
            return path.stat().st_size if path else 0
        except OSError:
            return 0

    encoder = models_dir() / "text_encoders" / "MiniMax-H3-text-encoder"
    encoder_bytes = sum(f.stat().st_size for f in encoder.rglob("*") if f.is_file()) if (
        encoder.is_dir()
    ) else 0
    diffusion = size(resolve_transformer(partition))
    if factorised:
        diffusion = int(diffusion * (1 - ADALN_SHARE))
    return {
        "diffusion_bytes": diffusion,
        "text_encoder_bytes": encoder_bytes,
        "vae_bytes": size(resolve("vae", VIDEO_VAE_FILE)) + size(resolve("vae", AUDIO_VAE_FILE)),
    }
