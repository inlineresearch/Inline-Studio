"""The installed-model catalog: what the user has dropped under the models root.

The root holds one subfolder per category (``diffusion_models``, ``vae``, ``loras``, ...). A scan
lists, per category, the weight files present (by filename) plus any subfolder that itself contains
weights (by folder name, e.g. a sharded ``qwen3-4b/`` text encoder). Non-weight files are ignored.

Two consumers: ``serialize.param_json`` fills a param's ``options_from`` select from ``list()``, and
the server folds ``fingerprint()`` into the registry version so dropping a weight in bumps it and
clients refetch ``/v1/models``. Nothing is downloaded; users place their own files.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import models_dirs

# Category subfolders scanned under the models root. These are the keys a param's `options_from`
# may reference (see graph/primitives.py); ensure_dirs() creates them so drop-in is obvious.
CATEGORIES: tuple[str, ...] = (
    "diffusion_models",
    "checkpoints",
    "vae",
    "text_encoders",
    "loras",
    "clip_vision",
    "controlnet",
    "upscale_models",
    # Latent-space upsamplers, kept apart from `upscale_models` (which is pixel-space) because they
    # are not interchangeable: LTX-2.5's two-stage pipeline upsamples between denoise stages.
    "latent_upscale_models",
    # Small weight patches applied to a loaded model rather than loaded as one, e.g. LTX-2.5's
    # duration head. Mirrors the folder the split packs publish them under.
    "model_patches",
    "embeddings",
    # Written by the preprocess runner rather than the user, but it holds real weights and the
    # panel is meant to show everything on disk.
    "annotators",
)

# Extensions we treat as model weights. A folder counts as a model if it contains one of these.
_WEIGHT_SUFFIXES: frozenset[str] = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
)


def _is_weight(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES


def _folder_has_weight(path: Path) -> bool:
    return any(_is_weight(child) for child in path.rglob("*"))


def _file_node(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "kind": "file",
        "sizeBytes": stat.st_size,
        "mtime": int(stat.st_mtime * 1000),
    }


def _dir_node(path: Path, name: str) -> dict[str, Any] | None:
    """A directory as a tree node, or None when it holds nothing worth showing."""
    if not path.is_dir():
        return None
    children: list[dict[str, Any]] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return None
    for entry in entries:
        # Caches like loras/.cache and text_encoders/.snap are noise in a panel about weights.
        if entry.name.startswith("."):
            continue
        if _is_weight(entry):
            children.append(_file_node(entry))
        elif entry.is_dir():
            child = _dir_node(entry, entry.name)
            if child is not None:
                children.append(child)
    if not children:
        return None
    return {
        "name": name,
        "path": str(path),
        "kind": "dir",
        "children": children,
        "fileCount": sum(1 for c in children if c["kind"] == "file"),
    }


class ModelCatalog:
    """Scans the models roots and answers "what's installed" per category.

    Cheap to construct; nothing touches disk until ``ensure_dirs`` or ``rescan``/``scan``. Results
    are cached so ``list`` and ``fingerprint`` are hits between scans. Several roots may be scanned
    (see ``config.models_dirs``); only the first is ever written to.
    """

    def __init__(self, root: Path | Sequence[Path]) -> None:
        roots = [Path(root)] if isinstance(root, str | Path) else [Path(r) for r in root]
        self._roots = roots or [Path("models")]
        self._entries: dict[str, list[str]] = {category: [] for category in CATEGORIES}

    @property
    def root(self) -> Path:
        """The writable root. Downloads and trained LoRAs land here."""
        return self._roots[0]

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def ensure_dirs(self) -> None:
        """Create the writable root and its category subfolders, so drop-in has somewhere to go."""
        for category in CATEGORIES:
            (self.root / category).mkdir(parents=True, exist_ok=True)

    def rescan(self) -> dict[str, list[str]]:
        """Re-read every category from every root, cache the result, and return it."""
        entries: dict[str, list[str]] = {}
        for category in CATEGORIES:
            names: list[str] = []
            for root in self._roots:
                for name in self._scan_category(root / category):
                    if name not in names:
                        names.append(name)
            entries[category] = sorted(names)
        self._entries = entries
        return entries

    def tree(self) -> list[dict[str, Any]]:
        """Every root as a nested, read-only listing for the Models panel.

        Separate from ``rescan`` because the select only needs flat names, while the panel wants
        nesting, sizes and mtimes; keeping them apart stops the hot path paying for the detail.
        """
        out: list[dict[str, Any]] = []
        for index, root in enumerate(self._roots):
            categories = [
                node
                for category in CATEGORIES
                if (node := _dir_node(root / category, category)) is not None
            ]
            out.append(
                {
                    "path": str(root),
                    "label": root.name or str(root),
                    "writable": index == 0,
                    "exists": root.is_dir(),
                    "categories": categories,
                }
            )
        return out

    # app.py calls scan() in the lifespan; rescan() is the same work exposed for tests/callers that
    # want the mapping back. Keep both so neither call site has to know about the other.
    def scan(self) -> dict[str, list[str]]:
        return self.rescan()

    def list(self, category: str) -> list[str]:
        """The installed entries for a category (empty for an unknown or empty one)."""
        return list(self._entries.get(category, []))

    def fingerprint(self) -> str:
        """A short, stable digest of the cached scan; changes iff the installed set changes."""
        payload = json.dumps(self._entries, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _scan_category(self, directory: Path) -> list[str]:
        if not directory.is_dir():
            return []
        names: list[str] = []
        for entry in directory.iterdir():
            if entry.name.startswith("."):
                continue
            if _is_weight(entry):
                names.append(entry.name)
            elif entry.is_dir() and _folder_has_weight(entry):
                # A sharded model (config + shards) is one entry, named for its folder.
                names.append(entry.name)
        return sorted(names)


def resolve_picked(category: str, chosen: object) -> Path | None:
    """A model dropdown's stored value, resolved to a file under ``category``.

    The select serves bare filenames, but projects written before it did store the full relative
    path, and joining one of those under the category doubles the prefix into a path that cannot
    exist. So: try the value as given under the category, then its bare name there, then the value
    as a path in its own right. Returns None when nothing matches, which callers must not stringify
    (``str(None)`` reaching a loader reads as a corrupt file rather than a stale pick).
    """
    name = str(chosen or "").strip()
    if not name:
        return None
    # Every root, writable one first, so a pick resolves wherever the user actually keeps it.
    for models_root in models_dirs():
        root = models_root / category
        for candidate in (root / name, root / Path(name).name):
            if candidate.exists():
                return candidate
    bare = Path(name)
    return bare if bare.exists() else None
