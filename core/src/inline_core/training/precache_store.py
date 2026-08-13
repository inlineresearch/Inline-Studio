"""Keep an encoded dataset on disk so a resumed or repeated run does not re-encode it.

The cache was in memory only, so every run paid the whole encode again. On a card that cannot hold
the text conditioner that pass runs on the CPU and takes hours, which means the recompute cost
landed hardest on the people least able to afford it.

Two things shape the design:

* **The dataset is re-exported per run**, into `training_runs/<id>/dataset`, so file times and paths
  differ between runs that hold identical images. The fingerprint is therefore the content of the
  files and the captions, never their paths or mtimes, and the cache lives outside the run folder.
* **Anything that changes what gets encoded is part of the key.** Resolution, flip, clip length and
  clip window all change the tensors, and a caption edit changes the conditioning. A stale hit is
  worse than a miss, because training would silently continue against the wrong latents.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("inline_core.training.precache")

#: Bumping this invalidates every cache written by an older layout.
#: 2: a Control run's reference is cached unbatched, matching its target. A cache written at 1 holds
#: five-dimensional references that the forward then makes six-dimensional.
FORMAT = "2"

#: Nested dicts on an item (H3 keeps an unconditional layout there) flatten onto one key with this.
_NEST = "//"


def fingerprint(dataset_dir: str, arch: str, settings: dict[str, Any]) -> str:
    """A key over the dataset's contents and everything that changes how it is encoded."""
    from . import dataset as ds

    parts: list[str] = [FORMAT, arch, json.dumps(settings, sort_keys=True)]
    pairs = ds.media_pairs(Path(dataset_dir))
    for path, caption in sorted(pairs, key=lambda p: p[0].name):
        parts.append(f"{path.name}:{path.stat().st_size}:{_content_hash(path)}:{caption}")
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load(
    root: Path, key: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float] | None:
    """The cached encode for ``key``, or None when there is not a complete one."""
    from safetensors.torch import load_file

    folder = root / key
    index = folder / "index.json"
    if not index.is_file():
        return None
    try:
        meta = json.loads(index.read_text(encoding="utf-8"))
        items = [_unflatten(load_file(str(folder / name))) for name in meta["items"]]
        uncond = _unflatten(load_file(str(folder / meta["unconditional"]))) if (
            meta.get("unconditional")
        ) else None
        shift = float(meta["shift"])
    except Exception as error:  # noqa: BLE001 - a damaged cache re-encodes rather than failing
        logger.warning("Ignoring an unreadable dataset cache at %s: %s", folder, error)
        return None
    return items, uncond, shift


def save(
    root: Path,
    key: str,
    items: list[dict[str, Any]],
    unconditional: dict[str, Any] | None,
    shift: float,
) -> None:
    """Write the encode for ``key``. Written to a temporary folder and renamed, so an interrupted
    write leaves no half-cache for the next run to trust."""
    from safetensors.torch import save_file

    folder = root / key
    if folder.exists():
        return
    staging = root / f".{key}.partial"
    try:
        staging.mkdir(parents=True, exist_ok=True)
        names = []
        for index, item in enumerate(items):
            name = f"item-{index:05d}.safetensors"
            save_file(_flatten(item), str(staging / name))
            names.append(name)
        # The shift is cached too: for the image archs it comes off a scheduler that a cache hit
        # never loads, so recomputing it is not an option.
        meta: dict[str, Any] = {"items": names, "unconditional": None, "shift": shift}
        if unconditional is not None:
            save_file(_flatten(unconditional), str(staging / "unconditional.safetensors"))
            meta["unconditional"] = "unconditional.safetensors"
        (staging / "index.json").write_text(json.dumps(meta), encoding="utf-8")
        staging.rename(folder)
    except Exception as error:  # noqa: BLE001 - caching is an optimisation, never a failure
        logger.warning("Could not write the dataset cache: %s", error)
        return
    logger.info("Cached the encoded dataset at %s", folder)


def _flatten(item: dict[str, Any]) -> dict[str, Any]:
    """One level of nesting onto flat keys, since safetensors stores a flat map of tensors.

    Non-tensor entries are dropped, which matches what the training loop does with them anyway.
    """
    out: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, dict):
            for inner, tensor in value.items():
                if hasattr(tensor, "contiguous"):
                    out[f"{key}{_NEST}{inner}"] = tensor.contiguous()
        elif hasattr(value, "contiguous"):
            out[key] = value.contiguous()
    return out


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        if _NEST in key:
            outer, inner = key.split(_NEST, 1)
            out.setdefault(outer, {})[inner] = value
        else:
            out[key] = value
    return out
