"""A cache for weights that were expensive to prepare, so the cost is paid once, not every launch.

Remapping a checkpoint's keys is a stream with renames and costs seconds. Quantising 60 GB of them
is minutes. Without somewhere to put the result, every cold start of a desktop app repeats it, so
this writes the **quantised** model out once and loads that thereafter. Deliberately not a cache of
the remapped bf16 weights: that would double the disk footprint to save the cheap half.

The identity is a hash of everything that could change the bytes - the source file, the key plan
version, the quantisation, the exclusion list, and any model-specific flag a caller adds. Miss that
and switching a flag serves a stale artifact, which looks exactly like the flag not working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import models_dir

logger = logging.getLogger("inline_core.prepared")

#: Dot-prefixed so it does not show up as an installed model in the catalog scan.
PREPARED_DIRNAME = ".prepared"
_MANIFEST = "prepared.json"


@dataclass(frozen=True)
class PreparedKey:
    """Everything that determines the prepared bytes."""

    source: Path
    plan_version: str
    quantization: str
    #: Model-specific switches: the AdaLN table flag, a derived-table identity, anything that
    #: changes the weights rather than how they are run.
    flags: Mapping[str, Any]

    def digest(self) -> str:
        stat = self.source.stat() if self.source.exists() else None
        payload = {
            "source": self.source.name,
            # Size and mtime rather than a content hash: hashing 60 GB to decide whether to skip
            # reading 60 GB defeats the point, and a re-downloaded file changes both.
            "size": stat.st_size if stat else 0,
            "mtime": int(stat.st_mtime) if stat else 0,
            "plan": self.plan_version,
            "quant": self.quantization,
            "flags": dict(sorted(self.flags.items())),
            "torchao": _torchao_version(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _torchao_version() -> str:
    """Part of the key: a torchao upgrade can change what its int8 tensors deserialise into."""
    try:
        import torchao

        return str(torchao.__version__)
    except Exception:  # noqa: BLE001 - absent torchao just means an unquantised artifact
        return "none"


def prepared_root() -> Path:
    return models_dir() / "diffusion_models" / PREPARED_DIRNAME


def prepared_dir(key: PreparedKey) -> Path:
    return prepared_root() / f"{key.source.stem}-{key.quantization}-{key.digest()}"


def lookup(key: PreparedKey) -> Path | None:
    """The cached artifact for this key, or None. A directory without its manifest is a crashed
    write and is treated as absent rather than loaded."""
    target = prepared_dir(key)
    if (target / _MANIFEST).is_file():
        logger.info("Prepared weights hit: %s", target.name)
        return target
    return None


def publish(key: PreparedKey, build: Any, *, describe: str = "") -> Path:
    """Run ``build(staging_dir)`` and move the result into place under this key.

    Staged then renamed, so an interrupted prepare never leaves something that looks finished. The
    manifest is written last for the same reason: it is what ``lookup`` treats as the commit.
    """
    target = prepared_dir(key)
    staging = target.with_name(target.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        build(staging)
        (staging / _MANIFEST).write_text(
            json.dumps(
                {
                    "source": str(key.source),
                    "planVersion": key.plan_version,
                    "quantization": key.quantization,
                    "flags": dict(key.flags),
                    "torchao": _torchao_version(),
                    "describe": describe,
                },
                indent=2,
            )
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    logger.info("Prepared weights written: %s (%s)", target.name, describe or key.quantization)
    return target


def reclaimable_source(key: PreparedKey) -> Path | None:
    """The original file a prepared artifact makes redundant, once that artifact exists.

    Surfaced rather than deleted: a user who will never run the full-precision build can reclaim
    tens of gigabytes, but that is their call, not ours.
    """
    if lookup(key) is None or not key.source.exists():
        return None
    return key.source


def prune(keep: set[Path] | None = None) -> int:
    """Delete prepared artifacts other than ``keep``. Returns how many went."""
    root = prepared_root()
    if not root.is_dir():
        return 0
    kept = keep or set()
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir() and entry not in kept:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed
