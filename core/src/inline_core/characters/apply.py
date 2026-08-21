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

    def __init__(
        self,
        name: str,
        refs: list[AssetRef],
        description: str,
        lora: Path | None = None,
        lora_strength: float = 1.0,
    ) -> None:
        self.name = name
        self.refs = refs
        self.description = description
        #: A trained adapter, which for a model with no reference channel is the only route.
        self.lora = lora
        #: What it fuses at. Set on Attach Adapter, because an overfit adapter is only usable
        #: turned down and the character wire carries no controls of its own.
        self.lora_strength = lora_strength

    def prompt_prefix(self, first_position: int) -> str:
        """Text naming the positions the character lands on, so ordinal prompting resolves."""
        if not self.refs:
            # A LoRA carries the likeness, so the description is all the prompt needs.
            detail = " ".join(self.description.split())
            return f"{detail} " if detail else ""
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


def char_apply(chosen: str, arch: str = encode.FLUX2_KLEIN_ARCH) -> AppliedCharacter | None:
    """How a character applies on ``arch``, or None when none is picked. An unreadable pick raises
    rather than silently generating the wrong person.

    ``arch`` matters because a model without a reference channel can only take the adapter, and its
    payloads are keyed separately."""
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
    references = arch == encode.FLUX2_KLEIN_ARCH

    if references and not cf.payload_valid(doc.manifest, arch, encode.PAYLOAD_ENCODER_VERSION):
        doc = _recompile(doc, path)
        digest = library.content_hash(path)

    description = _description(doc)
    lora = _extract_lora(doc, digest, arch)
    strength = encode.lora_strength(doc.manifest, arch)
    # A trained adapter wins unless the character says otherwise: the user asked for it explicitly,
    # and loading both would apply the identity twice.
    mode = doc.manifest.apply.get(arch) or ("lora" if lora else "reference")
    # No reference channel on this arch, so the adapter is the only way it can apply at all.
    if not references:
        mode = "lora"
    refs = [] if mode == "lora" else _extract(doc, digest, arch)
    return AppliedCharacter(
        doc.manifest.name or path.stem,
        refs,
        description,
        lora if mode == "lora" else None,
        strength,
    )


def _extract_lora(doc: cf.CharDoc, digest: str, arch: str) -> Path | None:
    """The adapter for ``arch``, or None. A stale one is the wrong face, so it is ignored."""
    entry = encode.lora_payload(doc.manifest, arch)
    if not entry:
        return None
    key = encode.payload_key(arch, encode.PAYLOAD_LORA)
    if not cf.payload_valid(doc.manifest, key, encode.LORA_PAYLOAD_VERSION):
        logger.info("Ignoring a stale %s adapter for %s", arch, doc.manifest.name)
        return None
    files = [str(f.get("path")) for f in entry.get("files") or [] if f.get("path")]
    if not files:
        return None
    root = _cache_root() / digest / key
    target = root / Path(files[0]).name
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        data = doc.members.get(files[0])
        if data is None:
            return None
        target.write_bytes(data)
    return target


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
