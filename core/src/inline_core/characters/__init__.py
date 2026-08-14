"""Persistent characters: the ``.char`` container, its compiled payloads, and identity scoring."""

from .charfile import (
    FORMAT_VERSION,
    MAGIC,
    SIGNATURE,
    CharDoc,
    CharFileError,
    Manifest,
    centroid_valid,
    looks_like_char,
    member_name,
    payload_valid,
    read,
    refs_fingerprint,
    sha256_bytes,
    write,
)

__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "SIGNATURE",
    "CharDoc",
    "CharFileError",
    "Manifest",
    "centroid_valid",
    "looks_like_char",
    "member_name",
    "payload_valid",
    "read",
    "refs_fingerprint",
    "sha256_bytes",
    "write",
]
