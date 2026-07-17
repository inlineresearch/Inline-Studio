"""Content-addressed input store: id = sha256 of the bytes, so re-uploading a file dedupes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..media import MediaKind

_KIND_BY_PREFIX = {
    "image/": MediaKind.IMAGE,
    "video/": MediaKind.VIDEO,
    "audio/": MediaKind.AUDIO,
}


@dataclass(frozen=True)
class StoredAsset:
    id: str
    kind: MediaKind
    size: int


class AssetStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def put(self, data: bytes, content_type: str | None) -> StoredAsset:
        self._root.mkdir(parents=True, exist_ok=True)
        asset_id = f"sha256-{hashlib.sha256(data).hexdigest()}"
        path = self._root / asset_id
        if not path.exists():
            path.write_bytes(data)
        return StoredAsset(id=asset_id, kind=_kind_for(content_type), size=len(data))

    def path(self, asset_id: str) -> Path | None:
        candidate = self._root / asset_id
        return candidate if candidate.exists() else None


def _kind_for(content_type: str | None) -> MediaKind:
    if content_type:
        for prefix, kind in _KIND_BY_PREFIX.items():
            if content_type.startswith(prefix):
                return kind
    return MediaKind.IMAGE
