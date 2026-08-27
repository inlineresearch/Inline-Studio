"""Checking a character's reference set before anything compiles a payload or trains on it.

Face identity only. A different outfit, setting or framing is exactly what the user was asked for,
so the subject term never decides whether a reference belongs - it only measures coverage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import charfile as cf
from . import encode, scoring

logger = logging.getLogger("inline_core.characters")

#: Verified against the character's own frozen gallery, or against nothing but itself.
MODE_EXISTING = "existing"
MODE_BOOTSTRAP = "bootstrap"


@dataclass
class Verdict:
    """What a pass found, with every list holding positions into ``manifest.refs``."""

    mode: str
    floor: float
    agreement: list[float | None] = field(default_factory=list)
    flagged: list[int] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    unchecked: list[int] = field(default_factory=list)
    #: Body and clothing references. Held out of scoring entirely rather than scored and excused:
    #: SFace measures faces, and a body shot that happens to show one would be judged on the wrong
    #: thing and could be flagged as an outlier for it.
    unscored: list[int] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "floor": self.floor,
            "agreement": self.agreement,
            "flagged": self.flagged,
            "duplicates": self.duplicates,
            "unchecked": self.unchecked,
            "unscored": self.unscored,
            "note": self.note,
        }


def duplicate_positions(manifest: cf.Manifest) -> list[int]:
    """Later copies of a reference already in the set, by content hash. No encoder needed.

    A duplicate is not a judgement call, and it is not harmless: it doubles that image's weight in
    a training mix and spends a reference slot the model addresses by position.
    """
    seen: set[str] = set()
    out: list[int] = []
    for index, ref in enumerate(manifest.refs):
        digest = str(ref.get("sha256") or "")
        if digest and digest in seen:
            out.append(index)
        seen.add(digest)
    return out


def verify(
    doc: cf.CharDoc,
    *,
    floor: float = scoring.REFERENCE_AGREEMENT_FLOOR,
    on_progress: encode.Progress | None = None,
) -> Verdict:
    """Score every reference against the character's identity, without changing anything."""
    report = on_progress or (lambda _fraction, _status: None)
    duplicates = duplicate_positions(doc.manifest)
    existing = encode.originals_frozen(doc.manifest)
    verdict = Verdict(mode=MODE_EXISTING if existing else MODE_BOOTSTRAP, floor=floor)
    verdict.duplicates = duplicates

    images = encode.ref_images(doc)
    total = len(images)
    faces = {
        i for i, ref in enumerate(doc.manifest.refs) if cf.role_of(ref) == cf.ROLE_FACE
    }
    verdict.unscored = [i for i in range(len(images)) if i not in faces]
    slots: list[list[float]] = []
    for index, image in enumerate(images):
        if index not in faces:
            slots.append([])
            continue
        report(0.1 + 0.6 * index / max(1, total), f"Checking reference {index + 1} of {total}…")
        slots.append(scoring.embed_face(image) or [])
    verdict.unchecked = [i for i, v in enumerate(slots) if not v and i in faces]

    live = [i for i in range(len(slots)) if i not in set(duplicates) and i in faces]
    if existing:
        verdict.agreement = _against_frozen(doc, slots, live)
    else:
        # Dedup first: a duplicate agrees with its twin at 100 and lifts the mean it is judged by.
        masked = [slots[i] if i in set(live) else [] for i in range(len(slots))]
        verdict.agreement = scoring.reference_agreement(masked)
        for index in duplicates:
            verdict.agreement[index] = None

    for index in verdict.unscored:
        if index < len(verdict.agreement):
            verdict.agreement[index] = None
    measured = [i for i, value in enumerate(verdict.agreement) if value is not None]
    if len(measured) < scoring.MIN_REFS_TO_FLAG:
        verdict.note = (
            f"{len(measured)} reference(s) with a usable face: below {scoring.MIN_REFS_TO_FLAG} "
            "there is no way to tell which one is the odd one out, so none were flagged."
        )
        return verdict
    verdict.flagged = [i for i in measured if (verdict.agreement[i] or 0.0) < floor]
    if verdict.unscored and not verdict.note:
        verdict.note = (
            f"{len(verdict.unscored)} body or clothing reference(s) are used but not scored: "
            "the face measure does not apply to them, and the subject measure conflates a body "
            "with its clothing and background."
        )
    return verdict


def _against_frozen(
    doc: cf.CharDoc, slots: list[list[float]], live: list[int]
) -> list[float | None]:
    """Each reference's mean similarity to the frozen originals, leaving out its own vector.

    Against the frozen gallery rather than the live set, so a harvested reference is measured by
    what the user vouched for and can never drift the target it is being measured against.
    """
    gallery, _subject = encode.frozen_originals(doc)
    frozen_paths = [str(r.get("path")) for r in doc.manifest.scoring["originals"]["refs"]]
    by_path = dict(zip(frozen_paths, gallery, strict=True))
    out: list[float | None] = []
    for index, vector in enumerate(slots):
        path = str(doc.manifest.refs[index].get("path") or "")
        if not vector or index not in set(live):
            out.append(None)
            continue
        others = [v for p, v in by_path.items() if p != path and v]
        out.append(scoring.agreement_against(vector, others))
    return out


def apply_verdict(doc: cf.CharDoc, verdict: Verdict, *, quarantine: bool) -> dict[str, list[str]]:
    """Act on a verdict. Duplicates always go; a flagged reference only when asked.

    Removed from the back, so a position still names the reference the report named.
    """
    before = [str(ref.get("path") or "") for ref in doc.manifest.refs]
    removed: dict[str, list[str]] = {"duplicates": [], "quarantined": []}
    targets = set(verdict.duplicates)
    if quarantine:
        targets |= set(verdict.flagged)
    for index in sorted(targets, reverse=True):
        if len(doc.manifest.refs) <= 1:
            logger.info("Keeping reference 1: a character needs at least one.")
            break
        if index in set(verdict.duplicates):
            # Byte-identical to one that stays, so there is nothing to preserve a copy of.
            encode.drop_ref(doc, index)
            removed["duplicates"].append(before[index])
        else:
            removed["quarantined"].append(encode.quarantine_ref(doc, index))
    _reindex(verdict, before, [str(ref.get("path") or "") for ref in doc.manifest.refs])
    doc.manifest.scoring["verification"] = {
        **verdict.to_json(),
        "checkedAt": int(time.time()),
        "removed": removed,
    }
    return removed


def _reindex(verdict: Verdict, before: list[str], after: list[str]) -> None:
    """Rewrite the verdict's positions onto the set that survived.

    Every list it holds is a position into `manifest.refs`, and a removal shifts each position
    after it - so a report stored as it was found would ring a reference that is now another one.
    """
    if before == after:
        return
    landed = {path: index for index, path in enumerate(after)}
    moved = {old: landed[path] for old, path in enumerate(before) if path in landed}
    verdict.agreement = [
        verdict.agreement[old] for old in sorted(moved, key=lambda old: moved[old])
    ]
    verdict.flagged = sorted(moved[i] for i in verdict.flagged if i in moved)
    verdict.unchecked = sorted(moved[i] for i in verdict.unchecked if i in moved)
    verdict.duplicates = sorted(moved[i] for i in verdict.duplicates if i in moved)
    verdict.unscored = sorted(moved[i] for i in verdict.unscored if i in moved)
