"""``char_encode``: reference images plus a description compiled into one ``.char``.

Refs go in untouched apart from EXIF rotation, because everything else is derived from them.
"""

from __future__ import annotations

import io
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import charfile as cf
from . import scoring

#: ``(fraction 0-1, human status)``. Reporting is best-effort - it never fails an encode.
Progress = Callable[[float, str], None]

#: The payload compiler's version. Bumping it recompiles every character's ref set on next use.
PAYLOAD_ENCODER_VERSION = "1"

#: Bumping this rebuilds every stored adapter, for a change in how one is trained or packed.
LORA_PAYLOAD_VERSION = "1"
PAYLOAD_ENCODER_ID = "flux2-klein-refset"
FLUX2_KLEIN_ARCH = "flux2-klein"

#: What a payload is, not which model it targets; one arch can carry both kinds.
PAYLOAD_REF = "ref"
PAYLOAD_LORA = "lora"


def payload_key(arch: str, kind: str = PAYLOAD_REF) -> str:
    """Where a payload lives. A reference set keeps the bare arch key, so v1 files stay valid."""
    return arch if kind == PAYLOAD_REF else f"{arch}-{kind}"

#: Done here so the pixels the pipeline sees are the pixels we hashed, once per character.
PAYLOAD_POLICY: dict[str, Any] = {"max_pixels": 1024 * 1024, "multiple_of": 16}

#: Enough pixels for SFace without carrying a full reference into the file.
FACE_CROP_SIZE = 512

_FACE_EMBEDS = f"scoring/embeds_{scoring.SFACE_ID}.json"
_SUBJECT_EMBEDS = f"scoring/embeds_{scoring.DINOV2_ID}.json"


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


def strength_hints(sizes: list[tuple[int, int]], framings: list[float] | None = None) -> list[str]:
    """One nudge per distinct gap. Creation is never gated on any of these.

    Takes sizes rather than images so it can be recomputed from a manifest, which is what keeps a
    rule change from leaving every existing character showing the advice it was encoded with.

    The full-body hint is the one that changes a number rather than taste: with no reference wider
    than a close-up, the subject term cannot speak to a wide take and is left out of the score
    entirely (``scoring.SUBJECT_FRAMING_RATIO``).
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
    if framings and not any(f <= scoring.WIDE_REF_FRACTION for f in framings):
        hints.append("Add a full-body shot, so wide takes can be scored")
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
    framings = [float(f) for f in (manifest.scoring.get("refFramings") or [])]
    return strength_hints(
        [(int(r.get("width") or 0), int(r.get("height") or 0)) for r in manifest.refs], framings
    )


def char_encode(
    refs: list[Path | str],
    *,
    name: str,
    description: str = "",
    app_version: str = "",
    char_id: str | None = None,
    created_at: int | None = None,
    on_progress: Progress | None = None,
) -> cf.CharDoc:
    """Compile reference images into a `CharDoc`. The caller writes it wherever it belongs.

    ``on_progress`` reports the encode as it runs. It matters because the first call on a machine
    downloads ~370MB of encoder weights, which without a signal is indistinguishable from a hang."""
    report = on_progress or (lambda _fraction, _status: None)
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

    report(0.05, "Reading references…")
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
    total = len(images)
    for index, image in enumerate(images):
        report(0.15 + 0.25 * index / total, f"Finding faces ({index + 1} of {total})…")
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

    report(0.4, "Normalising reference set…")
    build_payload(manifest, members, images)
    _build_centroids(manifest, members, images, crops, report)
    # From the manifest the scoring pass just wrote, so the hint and the score cannot disagree.
    manifest.hints = hints_for(manifest)
    report(1.0, "Done")
    return cf.CharDoc(manifest=manifest, members=members)


def append_refs(doc: cf.CharDoc, paths: list[Path]) -> None:
    """Add references without touching an encoder, so editing a character stays instant.

    Scoring and payloads go stale rather than being recomputed here; the fingerprint check already
    treats a changed reference set as invalid, and rebuilding is an explicit act."""
    used = {str(ref.get("path") or "") for ref in doc.manifest.refs}
    index = 0
    for path in paths:
        image = _open(path)
        while (member := cf.member_name("refs", index, ".png")) in used:
            index += 1
        used.add(member)
        data = _png_bytes(image)
        doc.members[member] = data
        doc.manifest.refs.append(
            {
                "path": member,
                "sha256": cf.sha256_bytes(data),
                "width": image.width,
                "height": image.height,
                "source_name": path.name,
            }
        )
    doc.manifest.modified_at = int(time.time())


def drop_ref(doc: cf.CharDoc, index: int) -> None:
    """Remove one reference, leaving a gap in the member numbering rather than renaming the rest."""
    if not 0 <= index < len(doc.manifest.refs):
        raise ValueError("That reference is not in this character.")
    if len(doc.manifest.refs) == 1:
        raise ValueError("A character needs at least one reference image.")
    removed = doc.manifest.refs.pop(index)
    member = str(removed.get("path") or "")
    doc.members.pop(member, None)
    # Its face crop was derived from it, so it goes too rather than pointing at a missing ref.
    for entry in [d for d in doc.manifest.derived if str(d.get("from") or "") == member]:
        doc.members.pop(str(entry.get("path") or ""), None)
        doc.manifest.derived.remove(entry)
    doc.manifest.modified_at = int(time.time())


def needs_rebuild(manifest: cf.Manifest) -> bool:
    """Whether the reference set has moved on from what the payload and scoring were built from."""
    entry = manifest.payloads.get(FLUX2_KLEIN_ARCH)
    if not isinstance(entry, dict):
        return bool(manifest.refs)
    return str(entry.get("source_sha256") or "") != cf.refs_fingerprint(manifest, PAYLOAD_POLICY)


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
        "type": PAYLOAD_REF,
        "encoder": {"id": PAYLOAD_ENCODER_ID, "version": PAYLOAD_ENCODER_VERSION},
        "source_sha256": cf.refs_fingerprint(manifest, PAYLOAD_POLICY),
        "policy": dict(PAYLOAD_POLICY),
        "files": files,
    }


def set_lora_payload(
    manifest: cf.Manifest,
    members: dict[str, bytes],
    adapter: bytes,
    *,
    arch: str = FLUX2_KLEIN_ARCH,
    base: str,
    rank: int,
    steps: int,
    resolution: int,
) -> str:
    """Store a trained adapter as this character's LoRA payload for ``arch``.

    Records what it was trained against, because a LoRA is only valid for that base: loading a 4B
    adapter onto a 9B silently degrades rather than raising. Shares the reference fingerprint, so
    editing the reference set invalidates the adapter the same way it invalidates a reference set.
    """
    key = payload_key(arch, PAYLOAD_LORA)
    for stale in [m for m in members if m.startswith(f"payloads/{key}/")]:
        members.pop(stale, None)
    member = f"payloads/{key}/adapter.safetensors"
    members[member] = adapter
    manifest.payloads[key] = {
        "payload_version": 1,
        "type": PAYLOAD_LORA,
        "encoder": {"id": f"{arch}-lora", "version": LORA_PAYLOAD_VERSION},
        "source_sha256": cf.refs_fingerprint(manifest, PAYLOAD_POLICY),
        "policy": dict(PAYLOAD_POLICY),
        "base": base,
        "training": {"rank": rank, "steps": steps, "resolution": resolution},
        "files": [{"path": member, "sha256": cf.sha256_bytes(adapter)}],
    }
    return key


def lora_payload(manifest: cf.Manifest, arch: str = FLUX2_KLEIN_ARCH) -> dict[str, Any] | None:
    """This character's adapter for ``arch``, or None when it has not been trained one."""
    entry = manifest.payloads.get(payload_key(arch, PAYLOAD_LORA))
    return entry if isinstance(entry, dict) else None


def _build_centroids(
    manifest: cf.Manifest,
    members: dict[str, bytes],
    images: list[Any],
    crops: list[Any | None],
    report: Progress = lambda _fraction, _status: None,
) -> None:
    centroids: dict[str, str] = {}

    report(0.55, "Loading identity encoders…")
    # Whole frame, not the crop: SFace self-aligns, and mismatching the two sides costs ~10 points.
    face_vectors = [v for image in images if (v := scoring.embed_face(image))]
    face_centroid = scoring.mean_vector(face_vectors)
    if face_centroid:
        member = f"scoring/centroid_{scoring.SFACE_ID}.json"
        members[member] = scoring.dump_centroid(face_centroid, len(face_vectors))
        centroids[scoring.SFACE_ID] = member
        # Every view kept, not just their mean: the face term matches the best-fitting one.
        members[_FACE_EMBEDS] = scoring.dump_embeds(face_vectors)

    report(0.7, "Measuring the subject…")
    subject_vectors = [v for image in images if (v := scoring.embed_subject(image))]
    subject_centroid = scoring.mean_vector(subject_vectors)
    if subject_centroid:
        member = f"scoring/centroid_{scoring.DINOV2_ID}.json"
        members[member] = scoring.dump_centroid(subject_centroid, len(subject_vectors))
        centroids[scoring.DINOV2_ID] = member
        # Keep every view: a mean over chest-up refs matches none of them.
        members[_SUBJECT_EMBEDS] = scoring.dump_embeds(subject_vectors)

    report(0.9, "Measuring reference coverage…")
    # How wide each reference is, so scoring can tell whether the gallery covers a take's framing.
    framings = [f for image in images if (f := scoring.face_fraction(image)) is not None]

    manifest.scoring = {
        "encoders": scoring.encoder_versions(),
        "centroids": centroids,
        "faceEmbeds": _FACE_EMBEDS if face_centroid else "",
        "subjectEmbeds": _SUBJECT_EMBEDS if subject_centroid else "",
        "refFramings": framings,
        "refAgreement": scoring.reference_agreement(face_vectors),
        "flaggedRefs": scoring.flagged_references(face_vectors),
        "face_bearing": bool(face_centroid),
        "blend": {"face": scoring.FACE_WEIGHT, "subject": scoring.SUBJECT_WEIGHT},
    }
