"""Reading somebody else's dataset folder: captions, and which clip pairs with which.

A published IC-LoRA set carries its pairing in a sidecar file rather than in filenames, and the two
conventions in circulation disagree on both the container and the field names. Lightricks ship a
``dataset.json`` holding one JSON array; the Hugging Face datasets convention is a
``metadata.jsonl`` of one object per line keyed on ``file_name``. Neither is guessable from the
other, so both are read and the caption key is discovered rather than assumed - the pixel-art set
names it ``caption-nvila15b``.

Filenames remain the fallback: ``bear.mp4`` pairs with ``bear_reference.mp4`` with no sidecar at
all, which is what ``training/dataset.py`` already reads at train time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Checked in order; the first that parses wins.
_METADATA_FILES = ("dataset.json", "metadata.jsonl", "dataset.jsonl", "metadata.json")

#: Exact keys first, then any key containing "caption", so `caption-nvila15b` and friends are found
#: without listing every captioner anyone has ever run.
_CAPTION_KEYS = ("caption", "text", "prompt", "description")
_TARGET_KEYS = ("media_path", "file_name", "path", "video", "image")
_REFERENCE_KEYS = ("reference_path", "reference", "control_path", "control")


@dataclass(frozen=True)
class DatasetEntry:
    """One row of a dataset's metadata, with paths still relative to the folder."""

    target: str
    caption: str
    reference: str | None


def read_metadata(root: Path) -> dict[str, DatasetEntry]:
    """Every metadata row, keyed by target filename. Empty when the folder carries none."""
    for name in _METADATA_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            rows = _rows(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # a malformed sidecar falls back to filenames rather than failing the import
        entries = {}
        for row in rows:
            target = _first(row, _TARGET_KEYS)
            if not target:
                continue
            entries[Path(target).name] = DatasetEntry(
                target=Path(target).name,
                caption=_caption(row),
                reference=(lambda r: Path(r).name if r else None)(_first(row, _REFERENCE_KEYS)),
            )
        if entries:
            return entries
    return {}


def _rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl" or text[0] != "[":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    return loaded if isinstance(loaded, list) else [loaded]


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _caption(row: dict[str, Any]) -> str:
    exact = _first(row, _CAPTION_KEYS)
    if exact:
        return exact
    for key, value in row.items():
        if "caption" in key.lower() and isinstance(value, str) and value.strip():
            return value.strip()
    return ""


#: What a training dataset can hold. Kept here rather than imported from `assets` so this module
#: stays a plain reader with no Studio dependencies.
_MEDIA_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
)


@dataclass(frozen=True)
class RepoPreview:
    """What pulling a Hugging Face dataset would get, answered without pulling it."""

    repo: str
    items: int
    pairs: int
    bytes: int
    metadata_file: str | None
    #: Set when the repo is unusable, so the caller can say why rather than showing an empty count.
    problem: str | None = None


def inspect_repo(repo: str, token: str | None = None) -> RepoPreview:
    """List a dataset repo's files and work out what it would import.

    Listing only: a video dataset is tens of gigabytes and the point of the preview is to let
    someone decline before paying for it. Pairing is read off filenames here rather than the
    metadata sidecar, which would need a download; the real pairing happens on import, where the
    sidecar is present and wins.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True, token=token)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401 or status == 403:
            gated = "That dataset is gated; accept its terms on Hugging Face first."
            return RepoPreview(repo, 0, 0, 0, None, gated)
        if status == 404:
            return RepoPreview(repo, 0, 0, 0, None, f"No dataset repo called {repo!r}.")
        return RepoPreview(repo, 0, 0, 0, None, str(exc).splitlines()[0])

    names, total = [], 0
    metadata_file = None
    for sibling in info.siblings or []:
        name = Path(sibling.rfilename).name
        if sibling.rfilename in _METADATA_FILES or name in _METADATA_FILES:
            metadata_file = name
        if Path(name).suffix.lower() in _MEDIA_SUFFIXES:
            names.append(name)
            total += int(getattr(sibling, "size", 0) or 0)

    return _preview(repo, names, total, metadata_file)


def inspect_folder(path: str) -> RepoPreview:
    """The same preview for a folder on this machine, so Path and Hugging Face behave alike."""
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return RepoPreview(path, 0, 0, 0, None, f"No folder at {path!r}.")
    names, total = [], 0
    metadata_file = None
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in _METADATA_FILES:
            metadata_file = entry.name
        if entry.suffix.lower() in _MEDIA_SUFFIXES:
            names.append(entry.name)
            total += entry.stat().st_size
    return _preview(path, names, total, metadata_file)


def _preview(
    label: str, names: list[str], total: int, metadata_file: str | None
) -> RepoPreview:
    present = set(names)
    references = {n for n in names if _looks_like_reference(n)}
    items = [n for n in names if n not in references]
    pairs = sum(1 for n in items if _reference_name_for(n, present))
    if not items:
        return RepoPreview(label, 0, 0, 0, metadata_file, "No images or clips there.")
    return RepoPreview(label, len(items), pairs, total, metadata_file)


def _looks_like_reference(name: str) -> bool:
    stem = Path(name).stem
    return stem.endswith("_reference") or stem.endswith(".ref")


def _reference_name_for(name: str, present: set[str]) -> str | None:
    path = Path(name)
    for suffix in (".ref", "_reference"):
        candidate = f"{path.stem}{suffix}{path.suffix.lower()}"
        if candidate in present:
            return candidate
    return None
