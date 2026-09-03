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

#: What each model accepts as a reference. A video model has its own frame grid, so the policy
#: cannot stay one constant - and it rides in the payload entry, because the fingerprint is taken
#: against the policy the payload was built with, not against whatever is current.
#: H3 resizes a reference to a 2048 short edge, upscaling included, rounding each axis to 32 on its
#: own, with no area cap - so a 4:1 reference is 8192x2048 (vendor/packing_ref2va.py).
MINIMAX_H3_ARCH = "minimax-h3"
#: H3 encodes a reference at a 2048 short edge, which is 4096 vision tokens each through Qwen3-VL.
MINIMAX_H3_SHORT_EDGE = 2048
MINIMAX_H3_POLICY: dict[str, Any] = {
    "short_edge": MINIMAX_H3_SHORT_EDGE,
    "multiple_of": 32,
    "max_aspect": 4.0,
}

#: One profile for every hosted fal endpoint, because they want the same pixels: they declare no
#: frame grid, and what actually bounds a reference is the wire, not the model - a reference travels
#: base64 from Core to the browser, back to Core, and on to fal.
FAL_REF_ARCH = "fal-ref"
FAL_REF_POLICY: dict[str, Any] = {"max_pixels": 1024 * 1024, "multiple_of": 8}

REFERENCE_POLICIES: dict[str, dict[str, Any]] = {
    FLUX2_KLEIN_ARCH: PAYLOAD_POLICY,
    MINIMAX_H3_ARCH: MINIMAX_H3_POLICY,
    FAL_REF_ARCH: FAL_REF_POLICY,
}


#: How the slots divide when there are more references than a model takes: face gets half, body and
#: cloth a quarter each. Face is weighted because it is what identity is actually carried by; the
#: other two are conditioning on top of it.
ROLE_RATIO: dict[str, int] = {cf.ROLE_FACE: 2, cf.ROLE_BODY: 1, cf.ROLE_CLOTH: 1}


def allocate_roles(counts: dict[str, int], cap: int) -> dict[str, int]:
    """How many of each role to send, capped at ``cap`` total.

    Under the cap everything goes. Over it, the split is `ROLE_RATIO` by largest remainder, and
    then any slot a role cannot fill is handed to the roles that still have references waiting -
    a character with no body shots should not lose those slots to nothing.
    """
    wanted = {role: max(0, int(counts.get(role, 0))) for role in cf.ROLES}
    if sum(wanted.values()) <= cap:
        return wanted

    total_weight = sum(ROLE_RATIO.values())
    exact = {role: cap * ROLE_RATIO[role] / total_weight for role in cf.ROLES}
    share = {role: min(wanted[role], int(exact[role])) for role in cf.ROLES}

    # Largest remainder first, then whoever still has references left, so no slot goes unused.
    spare = cap - sum(share.values())
    order = sorted(cf.ROLES, key=lambda r: (-(exact[r] - int(exact[r])), -ROLE_RATIO[r], r))
    while spare > 0:
        moved = False
        for role in order:
            if spare and share[role] < wanted[role]:
                share[role] += 1
                spare -= 1
                moved = True
        if not moved:
            break
    return share


def reference_policy(arch: str) -> dict[str, Any]:
    return REFERENCE_POLICIES.get(arch, PAYLOAD_POLICY)


#: What "Stored Reference Resolution" means when left at -1: the model's own policy, uncapped.
NO_REFERENCE_CAP = -1


def capped_policy(arch: str, resolution: int | None) -> dict[str, Any]:
    """A model's reference policy, with its target lowered to ``resolution``.

    This sets what the `.char` stores, not what a render costs. H3's pipeline calls
    `resolve_reference_image_size` on the way in and puts every reference back onto a 2048 short
    edge, upscaling included, so lowering this saves disk and compile time and no VRAM at all.
    """
    policy = dict(reference_policy(arch))
    if resolution is None or int(resolution) <= 0:
        return policy
    value = int(resolution)
    if "short_edge" in policy:
        policy["short_edge"] = min(int(policy["short_edge"]), value)
    if "max_pixels" in policy:
        policy["max_pixels"] = min(int(policy["max_pixels"]), value * value)
    return policy

#: Enough pixels for SFace without carrying a full reference into the file.
FACE_CROP_SIZE = 512

#: 2: per-reference scoring lists are aligned with `manifest.refs`; a v1 file's are compacted, so
#: an index in one cannot be read as a reference position.
SCORING_VERSION = 2

ORIGIN_ORIGINAL = cf.ORIGIN_ORIGINAL
ORIGIN_HARVESTED = cf.ORIGIN_HARVESTED
origin_of = cf.origin_of

#: The harvested pool never outgrows the originals, so they stay at least half of the reference
#: payload and half of the training mix. Absolute cap on top, for a character with many originals.
MAX_HARVESTED = 12


def originals(manifest: cf.Manifest) -> list[dict[str, Any]]:
    return [ref for ref in manifest.refs if origin_of(ref) == ORIGIN_ORIGINAL]


def harvested(manifest: cf.Manifest) -> list[dict[str, Any]]:
    return [ref for ref in manifest.refs if origin_of(ref) == ORIGIN_HARVESTED]

_FACE_EMBEDS = f"scoring/embeds_{scoring.SFACE_ID}.json"
_SUBJECT_EMBEDS = f"scoring/embeds_{scoring.DINOV2_ID}.json"
_ORIGINALS_FACE = f"scoring/originals_{scoring.SFACE_ID}.json"
_ORIGINALS_SUBJECT = f"scoring/originals_{scoring.DINOV2_ID}.json"
_HARVEST_FACE = f"scoring/harvested_{scoring.SFACE_ID}.json"
_HARVEST_SUBJECT = f"scoring/harvested_{scoring.DINOV2_ID}.json"


def _open(path: Path) -> Any:
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        # Without EXIF rotation every downstream crop and centroid is wrong the same invisible way.
        return ImageOps.exif_transpose(handle).convert("RGB")


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def ref_images(doc: cf.CharDoc) -> list[Any]:
    """The character's own references decoded, so a rebuild reads truth not the user's library.

    One image per entry in `manifest.refs`, always. Skipping a missing member would shorten the
    list and silently shift every scoring position after it onto the wrong reference; a file whose
    bytes have gone is unrebuildable either way, so it is reported rather than worked around.
    """
    from PIL import Image

    out: list[Any] = []
    for ref in doc.manifest.refs:
        member = str(ref.get("path") or "")
        data = doc.members.get(member)
        if data is None:
            raise cf.CharFileError(f"Reference {member} is missing from this character.")
        with Image.open(io.BytesIO(data)) as handle:
            out.append(handle.convert("RGB").copy())
    return out


def normalise_reference(image: Any, policy: dict[str, Any] | None = None) -> Any:
    """A reference resized into a model's budget, on its grid, preserving aspect.

    Two policy shapes, because two models mean two things by "budget": `max_pixels` is an area cap
    that only ever shrinks, `short_edge` is a target the smaller side is scaled *onto*, up or down.
    """
    rules = policy or PAYLOAD_POLICY
    grid = int(rules["multiple_of"])
    width, height = image.width, image.height
    limit = rules.get("max_aspect")
    if limit and max(width / height, height / width) > float(limit):
        raise ValueError(
            f"A reference must be within 1:{limit:g} and {limit:g}:1 for this model, "
            f"got {width}x{height}."
        )
    if "short_edge" in rules:
        # Rounded, not floored: flooring a scaled-up edge can land back under the target.
        scale = int(rules["short_edge"]) / min(width, height)
        width = max(grid, round(width * scale / grid) * grid)
        height = max(grid, round(height * scale / grid) * grid)
    else:
        max_pixels = int(rules["max_pixels"])
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
    # Originals only, or harvesting three takes onto a one-reference character silences the very
    # hints - another angle, a profile, a full-body shot - the harvest pool depends on being met.
    return strength_hints(
        [(int(r.get("width") or 0), int(r.get("height") or 0)) for r in originals(manifest)],
        framings,
    )


def char_encode(
    refs: list[Path | str],
    *,
    name: str,
    roles: list[str] | None = None,
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
    # Face when unsaid, so a caller that predates roles writes exactly what it always did.
    tags = list(roles or [cf.ROLE_FACE] * len(paths))
    tags += [cf.ROLE_FACE] * (len(paths) - len(tags))
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
                "origin": cf.ORIGIN_ORIGINAL,
                "role": tags[index],
            }
        )

    text_member = "text/description.md"
    text_bytes = (description or "").encode("utf-8")
    members[text_member] = text_bytes
    manifest.text = {"path": text_member, "sha256": cf.sha256_bytes(text_bytes)}

    total = len(images)
    for index, image in enumerate(images):
        report(0.15 + 0.25 * index / total, f"Finding faces ({index + 1} of {total})…")
        crop = scoring.face_crop(image)
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
    _build_centroids(manifest, members, images, report)
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
                "origin": cf.ORIGIN_ORIGINAL,
            }
        )
    _invalidate_scoring(doc)
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
    # Or scoring keeps best-matching a take against the reference just deleted for being wrong.
    _invalidate_scoring(doc)
    doc.manifest.modified_at = int(time.time())


def rescore(doc: cf.CharDoc, on_progress: Progress | None = None) -> None:
    """Recompute the scoring block from the character's own references, in place.

    Never a re-encode: `char_encode` builds a fresh manifest and members, so rebuilding a stale
    character through it drops the trained adapter, every payload but flux2-klein, `apply` and
    `reserved` - and the write that follows is what puts that loss on disk.
    """
    report = on_progress or (lambda _fraction, _status: None)
    _build_centroids(doc.manifest, doc.members, ref_images(doc), report)
    doc.manifest.hints = hints_for(doc.manifest)
    report(1.0, "Done")
    doc.manifest.modified_at = int(time.time())


def _invalidate_scoring(doc: cf.CharDoc) -> None:
    """Drop scoring after the reference set changes; `_scoring_stale` then rebuilds it.

    Index surgery is not available: the stored per-reference lists are compacted, so a reference
    without a face already shifts every index after it and there is nothing to renumber against.
    """
    # The frozen identity survives: it is not derived from the set that just changed.
    doc.manifest.scoring = {
        key: value for key, value in doc.manifest.scoring.items() if key == "originals"
    }
    owned = ("scoring/centroid_", "scoring/embeds_")
    for member in [m for m in doc.members if m.startswith(owned)]:
        doc.members.pop(member, None)


def payload_stale(manifest: cf.Manifest, key: str) -> bool:
    """Whether one payload was built from references that have since changed.

    Checked against the policy stored *in that payload*, because two models normalise references
    differently and a fingerprint only means anything beside the policy that produced it.
    """
    entry = manifest.payloads.get(key)
    if not isinstance(entry, dict):
        return True
    policy = entry.get("policy") if isinstance(entry.get("policy"), dict) else PAYLOAD_POLICY
    return str(entry.get("source_sha256") or "") != cf.refs_fingerprint(manifest, policy)


def stale_payloads(manifest: cf.Manifest) -> list[str]:
    """Every compiled payload that no longer matches the references, newest format first."""
    return [key for key in manifest.payloads if payload_stale(manifest, key)]


def needs_rebuild(manifest: cf.Manifest) -> bool:
    """Whether anything compiled from the references is out of date."""
    if not manifest.payloads:
        return bool(manifest.refs)
    return bool(stale_payloads(manifest))


def build_payload(
    manifest: cf.Manifest,
    members: dict[str, bytes],
    images: list[Any],
    arch: str = FLUX2_KLEIN_ARCH,
    policy: dict[str, Any] | None = None,
) -> None:
    """(Re)compile one model's reference set. Public because ``apply`` rebuilds stale ones."""
    policy = policy or reference_policy(arch)
    for stale in [m for m in members if m.startswith(f"payloads/{arch}/")]:
        members.pop(stale, None)
    files: list[dict[str, Any]] = []
    order = _originals_first(manifest, len(images))
    for slot, index in enumerate(order):
        member = f"payloads/{arch}/ref_{slot:03d}.png"
        data = _png_bytes(normalise_reference(images[index], policy))
        members[member] = data
        # The role rides with the compiled file: `apply` needs it to number the prompt, and
        # recomputing the manifest order there would be the same rule written twice.
        role = cf.role_of(manifest.refs[index]) if index < len(manifest.refs) else cf.ROLE_FACE
        files.append({"path": member, "sha256": cf.sha256_bytes(data), "role": role})
    manifest.payloads[arch] = {
        "payload_version": 1,
        "type": PAYLOAD_REF,
        "encoder": {"id": PAYLOAD_ENCODER_ID, "version": PAYLOAD_ENCODER_VERSION},
        "source_sha256": cf.refs_fingerprint(manifest, policy),
        "policy": dict(policy),
        # The fingerprint covers originals only, so this is the only record of what else went in.
        "harvested_count": len(order) - len(originals(manifest)),
        "files": files,
    }


def _originals_first(manifest: cf.Manifest, count: int) -> list[int]:
    """Reference positions with originals ahead of harvested ones.

    Position is meaning - FLUX.2 addresses a reference by its number - so the ones the user
    vouched for take the leading slots. Sorted here rather than kept sorted in the manifest,
    because `append_refs` adds at the end and `drop_ref` refuses to renumber.
    """
    refs = manifest.refs[:count]
    ordered = [i for i, ref in enumerate(refs) if cf.origin_of(ref) == cf.ORIGIN_ORIGINAL]
    ordered += [i for i, ref in enumerate(refs) if cf.origin_of(ref) == cf.ORIGIN_HARVESTED]
    # Images that do not line up with the manifest keep their order, not one read off other refs.
    return ordered if len(ordered) == count else list(range(count))


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
    strength: float = 1.0,
) -> str:
    """Store a trained adapter as this character's LoRA payload for ``arch``.

    Records what it was trained against, because a LoRA is only valid for that base: loading a 4B
    adapter onto a 9B silently degrades rather than raising. Shares the reference fingerprint, so
    editing the reference set invalidates the adapter the same way it invalidates a reference set.

    ``strength`` rides with the adapter because an overfit one is only usable turned down, and the
    character applies through a wire that carries no controls of its own.
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
        "strength": float(strength),
        "training": {"rank": rank, "steps": steps, "resolution": resolution},
        "files": [{"path": member, "sha256": cf.sha256_bytes(adapter)}],
    }
    return key


def lora_strength(manifest: cf.Manifest, arch: str = FLUX2_KLEIN_ARCH) -> float:
    """The strength this character's adapter fuses at, or 1.0 for one filed before it was set."""
    entry = lora_payload(manifest, arch) or {}
    try:
        return float(entry.get("strength", 1.0))
    except (TypeError, ValueError):
        return 1.0


def lora_payload(manifest: cf.Manifest, arch: str = FLUX2_KLEIN_ARCH) -> dict[str, Any] | None:
    """This character's adapter for ``arch``, or None when it has not been trained one."""
    entry = manifest.payloads.get(payload_key(arch, PAYLOAD_LORA))
    return entry if isinstance(entry, dict) else None


def can_freeze(manifest: cf.Manifest) -> bool:
    """Whether there are enough originals for a frozen gallery to mean anything."""
    return len(originals(manifest)) >= scoring.MIN_REFS_TO_FLAG


def originals_frozen(manifest: cf.Manifest) -> bool:
    return bool((manifest.scoring.get("originals") or {}).get("refs"))


def originals_stale(manifest: cf.Manifest) -> bool:
    """Whether the frozen gallery's vectors were built by encoders that have since moved."""
    frozen = manifest.scoring.get("originals") or {}
    recorded = {str(e.get("id")): str(e.get("version")) for e in (frozen.get("encoders") or [])}
    return recorded != scoring.encoder_versions_by_id()


def freeze_originals(doc: cf.CharDoc) -> dict[str, Any]:
    """Establish the identity target: which references are the originals, and their embeddings.

    The frozen thing is the membership. The vectors beside it are cache keyed by encoder version,
    because cosine across two encoder builds is a number with no meaning while the pixels are
    still in the file - the same rule payloads already follow.
    """
    if not can_freeze(doc.manifest):
        raise ValueError(
            f"A character needs {scoring.MIN_REFS_TO_FLAG} original references before its identity "
            "can be frozen: below that, the odd one out cannot be told from the rest."
        )
    entries = originals(doc.manifest)
    doc.manifest.scoring["originals"] = {
        "refs": [
            {"path": str(ref.get("path") or ""), "sha256": str(ref.get("sha256") or "")}
            for ref in entries
        ],
        "vectors": _ORIGINALS_FACE,
        "subjectVectors": _ORIGINALS_SUBJECT,
        "encoders": scoring.encoder_versions(),
        "frozenAt": int(time.time()),
    }
    _embed_frozen(doc)
    return doc.manifest.scoring["originals"]


def frozen_originals(doc: cf.CharDoc) -> tuple[list[list[float]], list[list[float]]]:
    """The frozen gallery's face and subject vectors, re-embedded if the encoders have moved."""
    if not originals_frozen(doc.manifest):
        raise ValueError("This character has no frozen identity yet.")
    if originals_stale(doc.manifest):
        _embed_frozen(doc)
        doc.manifest.scoring["originals"]["encoders"] = scoring.encoder_versions()
    face = scoring.load_keyed(doc.members, _ORIGINALS_FACE)
    subject = scoring.load_keyed(doc.members, _ORIGINALS_SUBJECT)
    paths = [str(r.get("path")) for r in doc.manifest.scoring["originals"]["refs"]]
    return ([face.get(p) or [] for p in paths], [subject.get(p) or [] for p in paths])


def _embed_frozen(doc: cf.CharDoc) -> None:
    """(Re)build the frozen gallery's vectors from exactly the references it was frozen over."""
    from PIL import Image

    face: dict[str, list[float]] = {}
    subject: dict[str, list[float]] = {}
    for ref in doc.manifest.scoring["originals"]["refs"]:
        path = str(ref.get("path") or "")
        data = doc.members.get(path)
        if data is None:
            raise cf.CharFileError(f"The frozen reference {path} is missing, so identity is gone.")
        # A changed reference means the frozen set was tampered with; re-embedding would launder it.
        if cf.sha256_bytes(data) != str(ref.get("sha256") or ""):
            raise cf.CharFileError(f"The frozen reference {path} has changed since it was frozen.")
        with Image.open(io.BytesIO(data)) as handle:
            image = handle.convert("RGB").copy()
        if vector := scoring.embed_face(image):
            face[path] = vector
        if vector := scoring.embed_subject(image):
            subject[path] = vector
    doc.members[_ORIGINALS_FACE] = scoring.dump_keyed(face)
    doc.members[_ORIGINALS_SUBJECT] = scoring.dump_keyed(subject)


def quarantine_ref(doc: cf.CharDoc, index: int) -> str:
    """Take a reference out of the set but keep its bytes, so removal stays reversible.

    `drop_ref` pops the member outright, and refs are truth: the file it came from may be long
    gone and Write persists over the character with no undo.
    """
    if not 0 <= index < len(doc.manifest.refs):
        raise ValueError("That reference is not in this character.")
    data = doc.members.get(str(doc.manifest.refs[index].get("path") or ""))
    drop_ref(doc, index)
    if data is None:
        return ""
    member = cf.member_name("quarantined", _next_slot(doc, "quarantined"), ".png")
    doc.members[member] = data
    return member


def harvest_cap(manifest: cf.Manifest) -> int:
    """Never more harvested than originals, so the user's own references stay at least half."""
    return min(MAX_HARVESTED, len(originals(manifest)))


def add_harvested(
    doc: cf.CharDoc, image: Any, *, agreement: float | None, score: float, source_take: str = ""
) -> int:
    """Add an approved take to the harvested pool. Returns its position in `manifest.refs`."""
    member = cf.member_name("harvested", _next_slot(doc, "harvested"), ".png")
    data = _png_bytes(image)
    doc.members[member] = data
    doc.manifest.refs.append(
        {
            "path": member,
            "sha256": cf.sha256_bytes(data),
            "width": image.width,
            "height": image.height,
            "source_name": source_take or member,
            "origin": cf.ORIGIN_HARVESTED,
            "agreement": agreement,
            "score": score,
            "harvestedAt": int(time.time()),
            "sourceTake": source_take,
        }
    )
    pool_face = scoring.load_keyed(doc.members, _HARVEST_FACE)
    pool_subject = scoring.load_keyed(doc.members, _HARVEST_SUBJECT)
    if vector := scoring.embed_face(image):
        pool_face[member] = vector
    if vector := scoring.embed_subject(image):
        pool_subject[member] = vector
    doc.members[_HARVEST_FACE] = scoring.dump_keyed(pool_face)
    doc.members[_HARVEST_SUBJECT] = scoring.dump_keyed(pool_subject)
    _invalidate_scoring(doc)
    doc.manifest.modified_at = int(time.time())
    return len(doc.manifest.refs) - 1


def prune_harvested(doc: cf.CharDoc) -> list[str]:
    """Trim the pool to the cap, least distinctive first. Originals are never candidates.

    Coverage rather than score, because a pool of near-duplicates of the best-scoring angle is
    worth less to a compile or a train than one that spans the angles the originals miss.
    """
    removed: list[str] = []
    subject = scoring.load_keyed(doc.members, _HARVEST_SUBJECT)
    frozen = [v for v in scoring.load_keyed(doc.members, _ORIGINALS_SUBJECT).values() if v]
    while len(harvested(doc.manifest)) > harvest_cap(doc.manifest):
        pool = [
            (index, str(ref.get("path")))
            for index, ref in enumerate(doc.manifest.refs)
            if cf.origin_of(ref) == cf.ORIGIN_HARVESTED
        ]
        # Originals first in the list, so each candidate is measured against them as well as
        # against the rest of the pool; only the pool's own slice is a candidate for dropping.
        values = scoring.coverage_values(frozen + [subject.get(p) or [] for _i, p in pool])[
            len(frozen) :
        ]
        # Unmeasurable first: a candidate with no embedding has no coverage to argue for keeping it.
        worst = min(range(len(pool)), key=lambda n: (values[n] is not None, values[n] or 0.0))
        index, member = pool[worst]
        drop_ref(doc, index)
        subject.pop(member, None)
        removed.append(member)
    if removed:
        doc.members[_HARVEST_SUBJECT] = scoring.dump_keyed(subject)
        face = scoring.load_keyed(doc.members, _HARVEST_FACE)
        for member in removed:
            face.pop(member, None)
        doc.members[_HARVEST_FACE] = scoring.dump_keyed(face)
    return removed


def _next_slot(doc: cf.CharDoc, prefix: str) -> int:
    index = 0
    while cf.member_name(prefix, index, ".png") in doc.members:
        index += 1
    return index


def _build_centroids(
    manifest: cf.Manifest,
    members: dict[str, bytes],
    images: list[Any],
    report: Progress = lambda _fraction, _status: None,
) -> None:
    centroids: dict[str, str] = {}
    previous_scoring = dict(manifest.scoring)
    # Only what this function owns: the frozen originals and the harvested pool outlive a rescore.
    for stale in [m for m in members if m.startswith(("scoring/centroid_", "scoring/embeds_"))]:
        members.pop(stale, None)
    originals = [i for i, ref in enumerate(manifest.refs) if origin_of(ref) == ORIGIN_ORIGINAL]

    report(0.55, "Loading identity encoders…")
    # Whole frame, not the crop: SFace self-aligns, and mismatching the two sides costs ~10 points.
    # Aligned with `manifest.refs`, empty where no face was found, so a flag names a position.
    face_slots = [scoring.embed_face(image) or [] for image in images]
    face_vectors = [face_slots[i] for i in originals if face_slots[i]]
    face_centroid = scoring.mean_vector(face_vectors)
    if face_centroid:
        member = f"scoring/centroid_{scoring.SFACE_ID}.json"
        members[member] = scoring.dump_centroid(face_centroid, len(face_vectors))
        centroids[scoring.SFACE_ID] = member
        # Every view kept, not just their mean: the face term matches the best-fitting one.
        members[_FACE_EMBEDS] = scoring.dump_embeds(_masked(face_slots, originals))

    report(0.7, "Measuring the subject…")
    subject_slots = [scoring.embed_subject(image) or [] for image in images]
    subject_vectors = [subject_slots[i] for i in originals if subject_slots[i]]
    subject_centroid = scoring.mean_vector(subject_vectors)
    if subject_centroid:
        member = f"scoring/centroid_{scoring.DINOV2_ID}.json"
        members[member] = scoring.dump_centroid(subject_centroid, len(subject_vectors))
        centroids[scoring.DINOV2_ID] = member
        # Keep every view: a mean over chest-up refs matches none of them.
        members[_SUBJECT_EMBEDS] = scoring.dump_embeds(_masked(subject_slots, originals))

    report(0.9, "Measuring reference coverage…")
    # Compacted, not aligned: it is read as an unordered bag, and a null in it raises `hints_for`.
    framings = [
        f for i in originals if (f := scoring.face_fraction(images[i])) is not None
    ]

    manifest.scoring = {
        "encoders": scoring.encoder_versions(),
        # 2: per-reference lists are aligned with `manifest.refs`; a v1 file's are compacted.
        "version": SCORING_VERSION,
        # So adding or dropping a reference is detectable without running an encoder to find out.
        "refCount": len(manifest.refs),
        "centroids": centroids,
        "faceEmbeds": _FACE_EMBEDS if face_centroid else "",
        "subjectEmbeds": _SUBJECT_EMBEDS if subject_centroid else "",
        "refFramings": framings,
        "refAgreement": scoring.reference_agreement(face_slots),
        "flaggedRefs": scoring.flagged_references(face_slots),
        "face_bearing": bool(face_centroid),
        "blend": {"face": scoring.FACE_WEIGHT, "subject": scoring.SUBJECT_WEIGHT},
    }
    # Carried across, because neither is derived from the set this pass just measured.
    for key in ("originals", "harvested", "verification"):
        if key in previous_scoring:
            manifest.scoring[key] = previous_scoring[key]


def _masked(slots: list[list[float]], keep: list[int]) -> list[list[float]]:
    """The aligned slots with everything outside `keep` blanked.

    This is what keeps the take-scoring gallery originals-only while staying ref-aligned: `score`
    already drops empty vectors, so a blanked position simply is not a candidate to match against.
    """
    allowed = set(keep)
    return [vector if i in allowed else [] for i, vector in enumerate(slots)]
