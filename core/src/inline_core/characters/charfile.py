"""The ``.char`` container: a persistent character as one portable zip.

::

    Ada.char
      manifest.json           first member; the signature is at byte 0
      refs/000.png            user-provided reference images, immutable
      derived/face_000.png    auto-generated crops
      text/description.md     the locked character description
      payloads/<arch>/        compiled per model family, pure cache
      scoring/                identity centroids, one per scoring encoder

Refs and text are truth; payloads and centroids are cache, and must be regenerable from ``refs/``.
Torch-free and network-free, so a core install can read a character.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: A reader refuses anything above this rather than half-loading the wrong face.
FORMAT_VERSION = 1

MAGIC = "INLINECHAR"

#: Valid JSON *and* a signature, so a file is identifiable from 32 bytes without parsing.
SIGNATURE = f'{{"magic":"{MAGIC}","format_version":'.encode()

MANIFEST_NAME = "manifest.json"

#: Fixed so the signature lands at byte 0 and two writes produce identical bytes.
_KEY_ORDER = (
    "magic", "format_version", "char_id", "name", "created_at", "modified_at", "app",
    "app_version", "refs", "derived", "text", "payloads", "scoring", "hints", "apply", "reserved",
)


#: Where a reference came from. Nested inside `refs` rather than a new top-level key on purpose:
#: `from_json` models only the keys it knows, so a top-level addition is destroyed by any older
#: build that rewrites the file, and three code paths rewrite one routinely.
ORIGIN_ORIGINAL = "original"
ORIGIN_HARVESTED = "harvested"


def origin_of(ref: dict[str, Any]) -> str:
    """Absent means original: every reference written before harvesting existed is one."""
    return str(ref.get("origin") or ORIGIN_ORIGINAL)


#: What a reference is *of*. Sits beside `origin` for the same reason, and absent means face, so
#: every character written before roles existed keeps behaving exactly as it did.
ROLE_FACE = "face"
ROLE_BODY = "body"
ROLE_CLOTH = "cloth"
ROLES = (ROLE_FACE, ROLE_BODY, ROLE_CLOTH)


def role_of(ref: dict[str, Any]) -> str:
    """A reference's role, defaulting to face. An unknown value reads as face rather than raising:
    a manifest is user data, and refusing to open a character over one bad string helps nobody."""
    role = str(ref.get("role") or ROLE_FACE)
    return role if role in ROLES else ROLE_FACE


def by_role(refs: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """The references carrying one role, in manifest order."""
    return [ref for ref in refs if role_of(ref) == role]


class CharFileError(Exception):
    """A ``.char`` that cannot be trusted. The message is shown to the user."""


@dataclass
class Manifest:
    char_id: str
    name: str
    created_at: int
    modified_at: int
    app_version: str = ""
    refs: list[dict[str, Any]] = field(default_factory=list)
    derived: list[dict[str, Any]] = field(default_factory=list)
    text: dict[str, Any] = field(default_factory=dict)
    payloads: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)
    #: arch -> "reference" | "lora". Absent means "whatever this model does best", resolved at
    #: apply time, so an older file needs no migration.
    apply: dict[str, str] = field(default_factory=dict)
    reserved: dict[str, Any] = field(
        default_factory=lambda: {"adapters": {}, "video_payloads": {}, "members": []}
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "magic": MAGIC,
            "format_version": FORMAT_VERSION,
            "char_id": self.char_id,
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "app": "inline-studio",
            "app_version": self.app_version,
            "refs": self.refs,
            "derived": self.derived,
            "text": self.text,
            "payloads": self.payloads,
            "scoring": self.scoring,
            "hints": self.hints,
            "apply": self.apply,
            "reserved": self.reserved,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Manifest:
        return cls(
            char_id=str(raw.get("char_id", "")),
            name=str(raw.get("name", "")),
            created_at=int(raw.get("created_at", 0)),
            modified_at=int(raw.get("modified_at", 0)),
            app_version=str(raw.get("app_version", "")),
            refs=list(raw.get("refs") or []),
            derived=list(raw.get("derived") or []),
            text=dict(raw.get("text") or {}),
            payloads=dict(raw.get("payloads") or {}),
            scoring=dict(raw.get("scoring") or {}),
            hints=list(raw.get("hints") or []),
            apply={str(k): str(v) for k, v in (raw.get("apply") or {}).items()},
            reserved=dict(raw.get("reserved") or {}),
        )


@dataclass
class CharDoc:
    """A whole character in memory: the manifest plus every member's bytes."""

    manifest: Manifest
    members: dict[str, bytes] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def member_name(prefix: str, index: int, suffix: str) -> str:
    """``refs/000.png`` - always ASCII, since a user's filename may not extract on Windows."""
    return f"{prefix}/{index:03d}{suffix}"


def dumps_manifest(manifest: Manifest) -> bytes:
    """Manifest bytes with the signature at byte 0."""
    raw = manifest.to_json()
    ordered = {key: raw[key] for key in _KEY_ORDER if key in raw}
    data = json.dumps(ordered, ensure_ascii=True, separators=(",", ":")).encode()
    if not data.startswith(SIGNATURE):
        raise CharFileError("Character manifest lost its signature; the key order is wrong.")
    return data


def looks_like_char(data: bytes) -> bool:
    """Whether a manifest blob carries our signature. Cheap enough for an upload gate."""
    return data.startswith(SIGNATURE)


def read(path: Path | str) -> CharDoc:
    """Load a ``.char``. Raises `CharFileError` with user-facing copy on anything unreadable."""
    target = Path(path)
    try:
        with zipfile.ZipFile(target, "r") as archive:
            names = archive.namelist()
            if MANIFEST_NAME not in names:
                raise CharFileError(f"{target.name} is not a character file (no manifest).")
            raw = archive.read(MANIFEST_NAME)
            members = {
                n: archive.read(n)
                for n in names
                if n != MANIFEST_NAME and not n.endswith("/")
            }
    except zipfile.BadZipFile as error:
        raise CharFileError(f"{target.name} is not a readable character file.") from error

    if not looks_like_char(raw):
        raise CharFileError(f"{target.name} is not a character file (bad signature).")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CharFileError(f"{target.name} has a corrupt manifest.") from error

    version = int(parsed.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise CharFileError(
            f"{target.name} was written by a newer version of Inline Studio "
            f"(character format {version}, this build reads {FORMAT_VERSION}). Update to open it."
        )
    return CharDoc(manifest=Manifest.from_json(parsed), members=members)


def write(path: Path | str, doc: CharDoc) -> Path:
    """Write atomically. The archive closes before the replace, or Windows refuses it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".part")
    try:
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as archive:
            # Manifest first so a reader can identify the file from the front of the stream.
            archive.writestr(MANIFEST_NAME, dumps_manifest(doc.manifest))
            for name in sorted(doc.members):
                # PNGs are already compressed; deflating them again costs time and saves nothing.
                compress = zipfile.ZIP_STORED if name.endswith(".png") else zipfile.ZIP_DEFLATED
                archive.writestr(name, doc.members[name], compress_type=compress)
        os.replace(staging, target)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return target


def refs_fingerprint(manifest: Manifest, policy: dict[str, Any]) -> str:
    """Ordered ref hashes plus policy, so a reorder or a resize-budget change both invalidate.

    Originals only. A harvested reference in here would mark the trained adapter stale the moment
    one was taken, and `_extract_lora` drops a stale adapter with an INFO log - so the loop whose
    whole point is a better adapter would silently switch off the adapter the user already has.
    """
    kept = [ref for ref in manifest.refs if origin_of(ref) == ORIGIN_ORIGINAL]
    parts = [str(ref.get("sha256", "")) for ref in kept]
    parts.append(json.dumps(policy, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def refs_identity(manifest: Manifest) -> str:
    """Which reference set this is, order included and policy excluded.

    Separate from `refs_fingerprint`, which answers "is this payload still valid" and so covers
    only the originals and the policy. This answers "were two nodes handed the same character",
    which a payload node needs because it compiles from a doc it is not the one holding.
    """
    parts = [f"{ref.get('sha256', '')}:{origin_of(ref)}" for ref in manifest.refs]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def payload_valid(manifest: Manifest, arch: str, encoder_version: str) -> bool:
    """Whether ``payloads/<arch>/`` can be used as-is, or must be recompiled from ``refs/``."""
    payload = manifest.payloads.get(arch)
    if not isinstance(payload, dict):
        return False
    encoder = payload.get("encoder")
    if not isinstance(encoder, dict) or str(encoder.get("version", "")) != encoder_version:
        return False
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        return False
    return str(payload.get("source_sha256", "")) == refs_fingerprint(manifest, policy)


def centroid_valid(manifest: Manifest, encoder_id: str, encoder_version: str) -> bool:
    """Cosine similarity across two encoder builds is meaningless, so the version must match."""
    encoders = manifest.scoring.get("encoders")
    if not isinstance(encoders, list):
        return False
    for entry in encoders:
        if isinstance(entry, dict) and str(entry.get("id")) == encoder_id:
            return str(entry.get("version", "")) == encoder_version
    return False
