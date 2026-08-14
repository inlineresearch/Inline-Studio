"""``char_encode``: reference images plus a description compiled into one ``.char``.

Refs go in untouched apart from EXIF rotation, because everything else is derived from them.
"""

from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any

from . import charfile as cf
from . import scoring

#: The payload compiler's version. Bumping it recompiles every character's ref set on next use.
PAYLOAD_ENCODER_VERSION = "1"
PAYLOAD_ENCODER_ID = "flux2-klein-refset"
FLUX2_KLEIN_ARCH = "flux2-klein"

#: Done here so the pixels the pipeline sees are the pixels we hashed, once per character.
PAYLOAD_POLICY: dict[str, Any] = {"max_pixels": 1024 * 1024, "multiple_of": 16}

#: Enough pixels for SFace without carrying a full reference into the file.
FACE_CROP_SIZE = 512

_FACE_EMBEDS = f"scoring/embeds_{scoring.SFACE_ID}.json"


def _open(path: Path) -> Any:
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        # Without EXIF rotation every downstream crop and centroid is wrong the same invisible way.
        return ImageOps.exif_transpose(handle).convert("RGB")


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalise_reference(image: Any, policy: dict[str, Any] | None = None) -> Any:
    """A reference resized into FLUX.2's budget, on the grid, preserving aspect."""
    rules = policy or PAYLOAD_POLICY
    max_pixels = int(rules["max_pixels"])
    grid = int(rules["multiple_of"])
    width, height = image.width, image.height
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    width = max(grid, (width // grid) * grid)
    height = max(grid, (height // grid) * grid)
    if (width, height) == (image.width, image.height):
        return image
    from PIL import Image

    return image.resize((width, height), Image.LANCZOS)


def strength_hints(sizes: list[tuple[int, int]]) -> list[str]:
    """One nudge per distinct gap. Creation is never gated on any of these.

    Takes sizes rather than images so it can be recomputed from a manifest, which is what keeps a
    rule change from leaving every existing character showing the advice it was encoded with.
    """
    hints: list[str] = []
    # Escalating, never stacked: at one reference "add another angle" and "add a profile view" are
    # the same request worded twice, which reads as two chores instead of one.
    if len(sizes) == 1:
        hints.append("Add a second angle")
    elif len(sizes) == 2:
        hints.append("Add a profile view")
    elif len(sizes) < 4:
        hints.append("Add a different outfit or setting")
    ratios = [w / h for w, h in sizes if h]
    if len(sizes) >= 2 and ratios and abs(max(ratios) - min(ratios)) < 0.05:
        hints.append("Add a wider or tighter crop")
    return hints


def flags_for(doc: cf.CharDoc) -> dict[str, Any]:
    """Reference agreement recomputed from the stored embeddings, so a floor change reaches
    characters already on disk. Falls back to what was recorded at encode time."""
    refs = scoring.load_embeds(doc.members, str(doc.manifest.scoring.get("faceEmbeds") or ""))
    if not refs:
        return {
            "refAgreement": list(doc.manifest.scoring.get("refAgreement") or []),
            "flaggedRefs": list(doc.manifest.scoring.get("flaggedRefs") or []),
        }
    return {
        "refAgreement": scoring.reference_agreement(refs),
        "flaggedRefs": scoring.flagged_references(refs),
    }


def hints_for(manifest: cf.Manifest) -> list[str]:
    """Hints recomputed from the manifest, so they never go stale against the current rules."""
    return strength_hints(
        [(int(r.get("width") or 0), int(r.get("height") or 0)) for r in manifest.refs]
    )


def char_encode(
    refs: list[Path | str],
    *,
    name: str,
    description: str = "",
    app_version: str = "",
    char_id: str | None = None,
    created_at: int | None = None,
) -> cf.CharDoc:
    """Compile reference images into a `CharDoc`. The caller writes it wherever it belongs."""
    paths = [Path(p) for p in refs]
    if not paths:
        raise ValueError("A character needs at least one reference image.")
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"Reference image not found: {', '.join(missing)}")

    now = int(time.time())
    images = [_open(p) for p in paths]
    members: dict[str, bytes] = {}
    manifest = cf.Manifest(
        char_id=char_id or str(uuid.uuid4()),
        name=name.strip() or "Character",
        created_at=created_at or now,
        modified_at=now,
        app_version=app_version,
    )

    for index, (path, image) in enumerate(zip(paths, images, strict=True)):
        member = cf.member_name("refs", index, ".png")
        data = _png_bytes(image)
        members[member] = data
        manifest.refs.append(
            {
                "path": member,
                "sha256": cf.sha256_bytes(data),
                "width": image.width,
                "height": image.height,
                "source_name": path.name,
            }
        )

    text_member = "text/description.md"
    text_bytes = (description or "").encode("utf-8")
    members[text_member] = text_bytes
    manifest.text = {"path": text_member, "sha256": cf.sha256_bytes(text_bytes)}

    crops: list[Any | None] = []
    for index, image in enumerate(images):
        crop = scoring.face_crop(image)
        crops.append(crop)
        if crop is None:
            continue
        crop = crop.resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
        member = f"derived/face_{index:03d}.png"
        data = _png_bytes(crop)
        members[member] = data
        manifest.derived.append(
            {
                "path": member,
                "kind": "face_crop",
                "from": cf.member_name("refs", index, ".png"),
                "producer": "yunet",
                "producer_version": scoring.weights.YUNET_VERSION,
                "sha256": cf.sha256_bytes(data),
            }
        )

    build_payload(manifest, members, images)
    _build_centroids(manifest, members, images, crops)
    manifest.hints = strength_hints([(im.width, im.height) for im in images])
    return cf.CharDoc(manifest=manifest, members=members)


def build_payload(manifest: cf.Manifest, members: dict[str, bytes], images: list[Any]) -> None:
    """(Re)compile the flux2-klein reference set. Public because ``apply`` rebuilds stale ones."""
    for stale in [m for m in members if m.startswith(f"payloads/{FLUX2_KLEIN_ARCH}/")]:
        members.pop(stale, None)
    files: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        member = f"payloads/{FLUX2_KLEIN_ARCH}/ref_{index:03d}.png"
        data = _png_bytes(normalise_reference(image))
        members[member] = data
        files.append({"path": member, "sha256": cf.sha256_bytes(data)})
    manifest.payloads[FLUX2_KLEIN_ARCH] = {
        "payload_version": 1,
        "encoder": {"id": PAYLOAD_ENCODER_ID, "version": PAYLOAD_ENCODER_VERSION},
        "source_sha256": cf.refs_fingerprint(manifest, PAYLOAD_POLICY),
        "policy": dict(PAYLOAD_POLICY),
        "files": files,
    }


def _build_centroids(
    manifest: cf.Manifest,
    members: dict[str, bytes],
    images: list[Any],
    crops: list[Any | None],
) -> None:
    centroids: dict[str, str] = {}

    # Whole frame, not the crop: SFace self-aligns, and mismatching the two sides costs ~10 points.
    face_vectors = [v for image in images if (v := scoring.embed_face(image))]
    face_centroid = scoring.mean_vector(face_vectors)
    if face_centroid:
        member = f"scoring/centroid_{scoring.SFACE_ID}.json"
        members[member] = scoring.dump_centroid(face_centroid, len(face_vectors))
        centroids[scoring.SFACE_ID] = member
        # Every view kept, not just their mean: the face term matches the best-fitting one.
        members[_FACE_EMBEDS] = scoring.dump_embeds(face_vectors)

    subject_vectors = [v for image in images if (v := scoring.embed_subject(image))]
    subject_centroid = scoring.mean_vector(subject_vectors)
    if subject_centroid:
        member = f"scoring/centroid_{scoring.DINOV2_ID}.json"
        members[member] = scoring.dump_centroid(subject_centroid, len(subject_vectors))
        centroids[scoring.DINOV2_ID] = member

    manifest.scoring = {
        "encoders": scoring.encoder_versions(),
        "centroids": centroids,
        "faceEmbeds": _FACE_EMBEDS if face_centroid else "",
        "refAgreement": scoring.reference_agreement(face_vectors),
        "flaggedRefs": scoring.flagged_references(face_vectors),
        "face_bearing": bool(face_centroid),
        "blend": {"face": scoring.FACE_WEIGHT, "subject": scoring.SUBJECT_WEIGHT},
    }
