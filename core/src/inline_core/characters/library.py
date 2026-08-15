"""The character library: ``.char`` files in ``models/characters/``, global across projects.

A folder of files rather than a table, so a character stays portable and drop-in. Torch-free.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

from ..config import models_dir, models_dirs
from . import charfile as cf
from . import encode

CATEGORY = "characters"
SUFFIX = ".char"

_UNSAFE = re.compile(r"[^A-Za-z0-9 ._-]+")


def root() -> Path:
    """The writable characters folder, created on demand so a fresh install can save."""
    return models_dir() / CATEGORY


def file_name(name: str) -> str:
    """A safe filename; the real name lives in the manifest."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    cleaned = _UNSAFE.sub("", folded).strip().strip(".")
    return f"{cleaned or 'character'}{SUFFIX}"


def unique_name(name: str) -> str:
    """``Ada.char``, ``Ada-2.char``, ... so saving a second Ada never overwrites the first."""
    base = file_name(name)
    stem = base[: -len(SUFFIX)]
    candidate, index = base, 2
    while (root() / candidate).exists():
        candidate = f"{stem}-{index}{SUFFIX}"
        index += 1
    return candidate


def resolve(chosen: str) -> Path | None:
    """A dropdown value resolved to a file. Returns None rather than a guess; never stringify it."""
    name = str(chosen or "").strip()
    if not name:
        return None
    for models_root in models_dirs():
        base = models_root / CATEGORY
        for candidate in (base / name, base / Path(name).name):
            if candidate.is_file():
                return candidate
    return None


def list_files() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for models_root in models_dirs():
        base = models_root / CATEGORY
        if not base.is_dir():
            continue
        for entry in sorted(base.glob(f"*{SUFFIX}")):
            if entry.is_file() and entry.name not in seen:
                seen.add(entry.name)
                out.append(entry)
    return out


def summaries() -> list[dict[str, Any]]:
    """One row per character. An unreadable file is reported, not hidden."""
    rows: list[dict[str, Any]] = []
    for path in list_files():
        try:
            doc = cf.read(path)
        except cf.CharFileError as error:
            rows.append({"file": path.name, "name": path.stem, "error": str(error), "refs": 0})
            continue
        manifest = doc.manifest
        rows.append(
            {
                "file": path.name,
                "charId": manifest.char_id,
                "name": manifest.name or path.stem,
                "refs": len(manifest.refs),
                "createdAt": manifest.created_at,
                "modifiedAt": manifest.modified_at,
                "hints": encode.hints_for(manifest),
                "description": _description(doc),
                "sizeBytes": path.stat().st_size,
            }
        )
    return rows


def _description(doc: cf.CharDoc) -> str:
    path = str(doc.manifest.text.get("path") or "")
    raw = doc.members.get(path)
    return raw.decode("utf-8", errors="replace") if raw else ""


def content_hash(path: Path) -> str:
    """Byte hash, so an in-place edit is a different character to the cache and the extract dir."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(doc: cf.CharDoc, file: str | None = None) -> Path:
    target = root() / (file or unique_name(doc.manifest.name))
    return cf.write(target, doc)


def delete(file: str) -> bool:
    path = resolve(file)
    if path is None:
        return False
    path.unlink(missing_ok=True)
    return True


def import_bytes(data: bytes, suggested: str) -> Path:
    """Accept an uploaded ``.char``, validating it before it lands in the library."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "incoming.char"
        staged.write_bytes(data)
        doc = cf.read(staged)  # raises CharFileError on anything we would not be able to open later
    return save(doc, unique_name(doc.manifest.name or Path(suggested).stem))
