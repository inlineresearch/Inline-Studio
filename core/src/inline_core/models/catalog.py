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
from pathlib import Path

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
    "embeddings",
)

# Extensions we treat as model weights. A folder counts as a model if it contains one of these.
_WEIGHT_SUFFIXES: frozenset[str] = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
)


def _is_weight(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES


def _folder_has_weight(path: Path) -> bool:
    return any(_is_weight(child) for child in path.rglob("*"))


class ModelCatalog:
    """Scans the models root and answers "what's installed" per category.

    Cheap to construct; nothing touches disk until ``ensure_dirs`` or ``rescan``/``scan``. Results
    are cached so ``list`` and ``fingerprint`` are hits between scans.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._entries: dict[str, list[str]] = {category: [] for category in CATEGORIES}

    @property
    def root(self) -> Path:
        return self._root

    def ensure_dirs(self) -> None:
        """Create the root and every category subfolder, so users have somewhere to drop weights."""
        for category in CATEGORIES:
            (self._root / category).mkdir(parents=True, exist_ok=True)

    def rescan(self) -> dict[str, list[str]]:
        """Re-read every category from disk, cache the result, and return it."""
        entries: dict[str, list[str]] = {}
        for category in CATEGORIES:
            entries[category] = self._scan_category(self._root / category)
        self._entries = entries
        return entries

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
            if _is_weight(entry):
                names.append(entry.name)
            elif entry.is_dir() and _folder_has_weight(entry):
                # A sharded model (config + shards) is one entry, named for its folder.
                names.append(entry.name)
        return sorted(names)
