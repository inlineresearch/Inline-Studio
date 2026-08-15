"""``char_apply``: a character turned into the references and prompt text a run needs.

Payload PNGs are extracted from the zip once, keyed by content hash so an edit lands in a new
directory rather than mutating one a running graph is reading.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..config import data_dir
from ..takes import AssetRef
from . import charfile as cf
from . import encode, library

logger = logging.getLogger("inline_core.characters")

#: Extracted payload bytes to keep. Tiny next to a checkpoint, so this holds many characters.
_CACHE_LIMIT_BYTES = 2 * 1024**3


class AppliedCharacter:
    """What a runner needs: ordered reference handles, plus the prompt text that names them."""

    def __init__(self, name: str, refs: list[AssetRef], description: str) -> None:
        self.name = name
        self.refs = refs
        self.description = description

    def prompt_prefix(self, first_position: int) -> str:
        """Text naming the positions the character lands on, so ordinal prompting resolves."""
        if not self.refs:
            return ""
        positions = [str(first_position + i) for i in range(len(self.refs))]
        if len(positions) == 1:
            which = f"Image {positions[0]} shows"
        else:
            which = f"Images {', '.join(positions[:-1])} and {positions[-1]} show"
        line = f"{which} {self.name}, the same character in every image."
        detail = " ".join(self.description.split())
        if not detail:
            return f"{line} "
        # Prose, not comma tags: without this it runs into the user's prompt unpunctuated.
        if detail[-1] not in ".!?":
            detail += "."
        return f"{line} {detail} "


def _cache_root() -> Path:
    return data_dir() / "characters"


def char_apply(chosen: str) -> AppliedCharacter | None:
    """References and prompt text for a pick, or None when none. An unreadable pick raises rather
    than silently generating the wrong person."""
    name = str(chosen or "").strip()
    if not name:
        return None
    path = library.resolve(name)
    if path is None:
        raise FileNotFoundError(
            f"Character {name!r} is not in models/characters/. Pick another in the node's settings."
        )

    doc = cf.read(path)
    digest = library.content_hash(path)
    arch = encode.FLUX2_KLEIN_ARCH

    if not cf.payload_valid(doc.manifest, arch, encode.PAYLOAD_ENCODER_VERSION):
        doc = _recompile(doc, path)
        digest = library.content_hash(path)

    refs = _extract(doc, digest, arch)
    description = _description(doc)
    return AppliedCharacter(doc.manifest.name or path.stem, refs, description)


def _description(doc: cf.CharDoc) -> str:
    raw = doc.members.get(str(doc.manifest.text.get("path") or ""))
    return raw.decode("utf-8", errors="replace") if raw else ""


def _recompile(doc: cf.CharDoc, path: Path) -> cf.CharDoc:
    """Rebuild a stale payload from ``refs/``. Payloads are cache, so this always works."""
    logger.info("Recompiling the %s payload for %s", encode.FLUX2_KLEIN_ARCH, path.name)
    import io

    from PIL import Image

    images: list[Any] = []
    for ref in doc.manifest.refs:
        raw = doc.members.get(str(ref.get("path")))
        if raw is None:
            raise cf.CharFileError(
                f"{path.name} is missing reference {ref.get('path')}, so it cannot be rebuilt."
            )
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

    encode.build_payload(doc.manifest, doc.members, images)
    cf.write(path, doc)
    return doc


def _extract(doc: cf.CharDoc, digest: str, arch: str) -> list[AssetRef]:
    payload = doc.manifest.payloads.get(arch) or {}
    files = [str(entry.get("path")) for entry in payload.get("files") or []]
    if not files:
        return []

    root = _cache_root() / digest
    marker = root / ".complete"
    if not marker.is_file():
        staging = root.with_name(f"{digest}.part")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for member in files:
            data = doc.members.get(member)
            if data is None:
                raise cf.CharFileError(f"Character payload is missing {member}.")
            (staging / Path(member).name).write_bytes(data)
        (staging / ".complete").write_text("ok")
        shutil.rmtree(root, ignore_errors=True)
        staging.replace(root)
        _prune(_cache_root())

    return [AssetRef(ref="path", path=str(root / Path(member).name)) for member in files]


def _prune(root: Path) -> None:
    """Oldest-first eviction against a size cap, the same shape as the prompt-embed cache."""
    if not root.is_dir():
        return
    entries = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    total = 0
    for entry in entries:
        total += sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        if total > _CACHE_LIMIT_BYTES:
            shutil.rmtree(entry, ignore_errors=True)
