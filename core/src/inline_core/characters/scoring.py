"""Identity scoring: how close a generated image is to a character's reference centroid.

SFace is sharp on faces but blind without one; DINOv2 describes the whole subject but is blunt
between similar faces - so a face-bearing character blends them and anything else falls back to
the subject alone. CPU only: ``PIPELINES`` holds one model across every arch, and an encoder on the
accelerator after a render risks fragmenting the allocator for nothing.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..errors import ComponentError
from . import weights

logger = logging.getLogger("inline_core.characters")

#: Re-derived over 10 LFW identities (40 same-person, 2600 impostor pairs) once best-match fixed
#: the face term. Face carries nearly all the discrimination (same/impostor medians 76.8/13.7,
#: against 59.6/37.2 for subject) and separation rises with face weight all the way to 1.0. It
#: stops at 0.8 on purpose: the benchmark is face crops, so it cannot see a right face on a wrong
#: body, and subject is the only term that would.
FACE_WEIGHT = 0.8
SUBJECT_WEIGHT = 0.2

SFACE_ID = "sface"
DINOV2_ID = "dinov2-base"

#: Face identity is matched against the best-fitting reference, not their average. Views of one
#: person sit far apart in SFace space (~24% similar between a frontal and a flipped view), so a
#: centroid lands between them and matches none - which made every added reference lower the score.
#: DINOv2 keeps its centroid: its views agree to ~93%, so averaging denoises instead of blurring.

#: Below this a "face" is usually a texture, and scoring one is a confident number about nothing.
FACE_CONFIDENCE = 0.7

#: A reference whose mean SFace agreement with the rest of the set falls below this is flagged as
#: possibly a different person. Measured over LFW: a planted impostor scores p95=20.2 (max 26.1)
#: while a genuine reference scores p5=29.7 (median 52.0), so the floor sits in the empty band
#: between them - 98% of impostors caught for a 3% false-alarm rate, the same rate as a floor of 20.
#: Best-match scoring means one wrong reference is a backdoor, so this closes it at the source.
REFERENCE_AGREEMENT_FLOOR = 25.0

#: Below three references "mean agreement with the others" is a single pairwise number, which
#: cannot say which of the two is the odd one out.
MIN_REFS_TO_FLAG = 3

#: How far a take's framing may sit from the nearest reference's before the subject term is noise.
SUBJECT_FRAMING_RATIO = 2.5

#: A reference wider than a close-up: chest-up is 11-14% of frame, medium and wider below 5%.
WIDE_REF_FRACTION = 0.05

#: Square input YuNet and SFace expect after our own resize.
_DETECT_SIZE = 640

_models: dict[str, Any] = {}

#: Which annotator file backs each encoder, empty meaning the one `weights` ships with. Runs are
#: serialised on one worker, so a per-run selection needs no more than a module-level choice.
_chosen: dict[str, str] = {}


def use_encoders(
    face_detector: str = "", face_embedder: str = "", subject_embedder: str = ""
) -> None:
    """Choose the annotator files the encoders load. Empty keeps the shipped default."""
    _chosen.update(
        {"yunet": face_detector.strip(), "sface": face_embedder.strip(),
         "dinov2": subject_embedder.strip()}
    )


def use_encoders_from(scoring_block: dict[str, Any]) -> None:
    """Pin the encoders to the ones a character was scored with.

    `_chosen` is module state that nothing resets, so without this one Encode Character node run
    with a picked annotator marks every other character's centroid stale for the rest of the
    process - and each one is then rewritten on the next take it is scored against.
    """
    picked: dict[str, str] = {}
    for entry in scoring_block.get("encoders") or []:
        if not isinstance(entry, dict):
            continue
        version = str(entry.get("version") or "")
        name = version.split(":", 1)[1] if ":" in version else ""
        if str(entry.get("id")) == SFACE_ID:
            picked["sface"] = name
        elif str(entry.get("id")) == DINOV2_ID:
            picked["dinov2"] = name
    # The detector is not version-tracked, so it can only go back to the shipped default here.
    use_encoders(face_embedder=picked.get("sface", ""), subject_embedder=picked.get("dinov2", ""))


def chosen(kind: str, default: str) -> str:
    """The picked filename for an encoder, or the shipped one."""
    return _chosen.get(kind) or default


def _encoder_path(kind: str, default: str) -> Path:
    """Where an encoder loads from, and a clear error when a picked file is not there.

    `weights.ensure()` fetches the defaults; a file the user picked instead is theirs to provide,
    so it is reported rather than silently replaced by the default.
    """
    name = chosen(kind, default)
    path = weights.annotators_root() / name
    if name == default:
        weights.ensure()
        return path
    if not path.exists():
        raise ComponentError(f"{name} is not in models/annotators.")
    return path


def _yunet(width: int, height: int) -> Any:
    import cv2

    # Keyed by the file, so switching the pick loads the new one instead of serving the old.
    key = f"yunet:{chosen('yunet', weights.YUNET_FILE)}"
    detector = _models.get(key)
    if detector is None:
        path = _encoder_path("yunet", weights.YUNET_FILE)
        detector = cv2.FaceDetectorYN.create(str(path), "", (width, height), FACE_CONFIDENCE)
        _models[key] = detector
    detector.setInputSize((width, height))
    return detector


def _sface() -> Any:
    import cv2

    key = f"sface:{chosen('sface', weights.SFACE_FILE)}"
    model = _models.get(key)
    if model is None:
        model = cv2.FaceRecognizerSF.create(str(_encoder_path("sface", weights.SFACE_FILE)), "")
        _models[key] = model
    return model


def _dinov2() -> tuple[Any, Any]:
    key = f"dinov2:{chosen('dinov2', weights.DINOV2_DIR)}"
    pair = _models.get(key)
    if pair is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        root = str(_encoder_path("dinov2", weights.DINOV2_DIR))
        processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        model = AutoModel.from_pretrained(root, local_files_only=True, dtype=torch.float32)
        model.eval()
        pair = (processor, model)
        _models[key] = pair
    return pair


def unload() -> None:
    """Drop the cached encoders. Only needed by tests and by a models rescan."""
    _models.clear()


def _to_bgr(image: Any) -> Any:
    import numpy as np

    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def detect_face(image: Any) -> tuple[Any, float] | None:
    """The best face as YuNet's 15-value row (box + landmarks), which is what SFace aligns on."""
    bgr = _to_bgr(image)
    height, width = bgr.shape[:2]
    scale = _DETECT_SIZE / max(width, height)
    if scale < 1.0:
        import cv2

        bgr = cv2.resize(bgr, (int(width * scale), int(height * scale)))
        height, width = bgr.shape[:2]
    try:
        _, faces = _yunet(width, height).detect(bgr)
    except Exception as error:  # noqa: BLE001 - a detector failure must not break an encode
        logger.warning("Face detection failed: %s", error)
        return None
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda row: float(row[14]))
    return (bgr, best)


def face_crop(image: Any, margin: float = 0.35) -> Any | None:
    """A square crop around the detected face, in the original image's pixels, or None."""
    found = detect_face(image)
    if found is None:
        return None
    bgr, row = found
    detected_h, detected_w = bgr.shape[:2]
    scale_x = image.width / detected_w
    scale_y = image.height / detected_h
    x, y, w, h = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    cx, cy = (x + w / 2) * scale_x, (y + h / 2) * scale_y
    half = max(w * scale_x, h * scale_y) * (0.5 + margin)
    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(image.width, cx + half))
    bottom = int(min(image.height, cy + half))
    if right - left < 32 or bottom - top < 32:
        return None
    return image.convert("RGB").crop((left, top, right, bottom))


def embed_face(image: Any) -> list[float] | None:
    """A 128-d SFace embedding of the image's largest face, or None when there is no face."""
    found = detect_face(image)
    if found is None:
        return None
    bgr, row = found
    try:
        model = _sface()
        aligned = model.alignCrop(bgr, row)
        feature = model.feature(aligned)
    except Exception as error:  # noqa: BLE001 - never let scoring break the caller
        logger.warning("Face embedding failed: %s", error)
        return None
    return [float(v) for v in feature.reshape(-1)]


def embed_subject(image: Any) -> list[float] | None:
    """A 768-d DINOv2 CLS embedding of the whole image."""
    try:
        import torch

        processor, model = _dinov2()
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        vector = out.last_hidden_state[:, 0].reshape(-1)
    except Exception as error:  # noqa: BLE001 - never let scoring break the caller
        logger.warning("Subject embedding failed: %s", error)
        return None
    return [float(v) for v in vector]


def face_fraction(image: Any, found: Any | None = None) -> float | None:
    """The detected face's share of the frame - a cheap stand-in for how wide the shot is.

    Measured on the eval set: chest-up references land at 11-14%, close-ups at 8%, medium shots at
    3-5%, and full-body or wide shots below 4%. Pass ``found`` when the caller has already run the
    detector, so a take is never detected twice.
    """
    try:
        hit = detect_face(image) if found is None else found
        if hit is None:
            return None
        _bgr, row = hit
        area = float(row[2]) * float(row[3])
        frame = float(image.width * image.height)
        return area / frame if frame else None
    except Exception:  # noqa: BLE001 - a framing measure is never worth failing a score over
        return None


def framing_distance(query: float | None, references: list[float]) -> float | None:
    """How far a take's framing sits from the nearest reference's, as a ratio of 1.0 or more."""
    usable = [f for f in references if f]
    if not query or not usable:
        return None
    return min(max(f, query) / min(f, query) for f in usable)


def mean_vector(vectors: list[list[float]]) -> list[float] | None:
    """Centroid of unit-normalised vectors, so one high-magnitude ref cannot dominate."""
    usable = [normalise(v) for v in vectors if v]
    if not usable:
        return None
    width = len(usable[0])
    summed = [0.0] * width
    for vector in usable:
        for i in range(width):
            summed[i] += vector[i]
    return normalise([v / len(usable) for v in summed])


def normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else list(vector)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    left, right = normalise(a), normalise(b)
    return sum(x * y for x, y in zip(left, right, strict=True))


def to_percent(similarity: float) -> float:
    """Cosine to 0-100. Negative is no-match, not anti-match, so it clamps at zero."""
    return round(max(0.0, min(1.0, similarity)) * 100, 1)


def score(
    image: Any,
    centroids: dict[str, list[float]],
    face_refs: list[list[float]] | None = None,
    subject_refs: list[list[float]] | None = None,
    ref_framings: list[float] | None = None,
) -> dict[str, Any] | None:
    """One blended score, or None when nothing was measurable - which is not a zero.

    ``face_refs`` / ``subject_refs`` are the per-reference embeddings; without them a term falls
    back to its centroid, which is what a character encoded before those existed carries. Both
    terms match against the closest reference rather than a mean: averaging embeddings across views
    produces a centroid matching none of them.

    ``ref_framings`` decides whether the subject term is reported as trustworthy. When the gallery
    cannot cover this take's framing the subject number is noise, so it is excluded from the blend
    rather than blended in behind a confident-looking total.
    """
    face_centroid = centroids.get(SFACE_ID) or []
    subject_centroid = centroids.get(DINOV2_ID) or []

    face_score: float | None = None
    if face_centroid or face_refs:
        embedding = embed_face(image)
        if embedding:
            gallery = [v for v in (face_refs or []) if v]
            if not gallery and face_centroid:
                gallery = [face_centroid]
            if gallery:
                face_score = max(to_percent(cosine(embedding, ref)) for ref in gallery)

    subject_score: float | None = None
    if subject_centroid or subject_refs:
        embedding = embed_subject(image)
        if embedding:
            gallery = [v for v in (subject_refs or []) if v]
            if not gallery and subject_centroid:
                gallery = [subject_centroid]
            if gallery:
                subject_score = max(to_percent(cosine(embedding, ref)) for ref in gallery)

    # Counts unless there is positive evidence it cannot speak to this take; unknown is not bad.
    framings = [f for f in (ref_framings or []) if f]
    distance = framing_distance(face_fraction(image), framings)
    if distance is not None:
        covered = distance <= SUBJECT_FRAMING_RATIO
    else:
        # No face to measure is itself the wide shot a chest-up gallery cannot speak to.
        covered = not framings or any(f <= WIDE_REF_FRACTION for f in framings)
    subject_usable = subject_score is not None and covered

    if face_score is not None and subject_usable:
        blended = FACE_WEIGHT * face_score + SUBJECT_WEIGHT * (subject_score or 0.0)
    elif face_score is not None:
        blended = face_score
    elif subject_usable:
        blended = subject_score or 0.0
    else:
        return None

    return {
        "score": round(blended, 1),
        "faceScore": face_score,
        "subjectScore": subject_score,
        "faceBearing": face_score is not None,
        # False means the number above is face-only: the references cannot speak to this framing.
        "subjectCounted": subject_usable,
        "framingDistance": round(distance, 1) if distance is not None else None,
    }


def reference_agreement(face_refs: list[list[float]]) -> list[float | None]:
    """Each reference's mean SFace similarity to the others, 0-100, aligned with the input.

    `None` is a reference with no face to compare, which is not a low score. `cosine` against an
    empty vector is 0.0, so letting one into the average drags every genuine reference's mean
    toward the floor - four references with two wide shots among them would all fall under it.
    """
    measured = [i for i, vector in enumerate(face_refs) if vector]
    if len(measured) < 2:
        return [100.0 if vector else None for vector in face_refs]
    out: list[float | None] = []
    for i, vector in enumerate(face_refs):
        if not vector:
            out.append(None)
            continue
        others = [face_refs[j] for j in measured if j != i]
        out.append(round(sum(to_percent(cosine(vector, o)) for o in others) / len(others), 1))
    return out


def flagged_references(face_refs: list[list[float]]) -> list[int]:
    """Indices of references that may not be the same person. Face identity only - a different
    outfit or setting is exactly what the user was asked for and must not trip this."""
    if sum(1 for vector in face_refs if vector) < MIN_REFS_TO_FLAG:
        return []
    scores = reference_agreement(face_refs)
    return [i for i, s in enumerate(scores) if s is not None and s < REFERENCE_AGREEMENT_FLOOR]


def agreement_against(candidate: list[float], gallery: list[list[float]]) -> float | None:
    """A candidate's mean SFace similarity to a fixed gallery, or None when it cannot be measured.

    Separate from `reference_agreement` because the candidate is not a member of the gallery, so
    there is nothing to leave out. Below `MIN_REFS_TO_FLAG` genuine faces it declines to answer:
    against one or two references this is a pairwise number, and the floor is a mean over a set.
    """
    usable = [vector for vector in gallery if vector]
    if not candidate or len(usable) < MIN_REFS_TO_FLAG:
        return None
    return round(sum(to_percent(cosine(candidate, v)) for v in usable) / len(usable), 1)


def coverage_values(subject_refs: list[list[float]]) -> list[float | None]:
    """How much each reference shows that the others do not: 1 minus its closest cosine to them.

    DINOv2, never SFace: this measures framing and setting, which is the axis a gallery needs to
    span, and is exactly the axis that must never decide identity.
    """
    measured = [i for i, vector in enumerate(subject_refs) if vector]
    if len(measured) < 2:
        return [1.0 if vector else None for vector in subject_refs]
    out: list[float | None] = []
    for i, vector in enumerate(subject_refs):
        if not vector:
            out.append(None)
            continue
        others = [subject_refs[j] for j in measured if j != i]
        out.append(round(1.0 - max(cosine(vector, o) for o in others), 4))
    return out


def load_centroids(members: dict[str, bytes], paths: dict[str, Any]) -> dict[str, list[float]]:
    """Centroid vectors out of a ``.char``'s members, keyed by encoder id."""
    import json

    out: dict[str, list[float]] = {}
    for encoder_id, member in paths.items():
        raw = members.get(str(member))
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            vector = [float(v) for v in parsed.get("vector") or []]
        except (ValueError, TypeError, AttributeError):
            continue
        if vector:
            out[str(encoder_id)] = vector
    return out


def dump_centroid(vector: list[float], count: int) -> bytes:
    import json

    return json.dumps({"vector": vector, "count": count}, separators=(",", ":")).encode()


def dump_keyed(vectors: dict[str, list[float]]) -> bytes:
    import json

    return json.dumps({"vectors": vectors}, separators=(",", ":")).encode()


def load_keyed(members: dict[str, bytes], path: str) -> dict[str, list[float]]:
    """Vectors keyed by member name, for a pool where a position is not a stable identifier."""
    import json

    raw = members.get(str(path))
    if not raw:
        return {}
    try:
        parsed = (json.loads(raw).get("vectors") or {}).items()
    except (ValueError, TypeError, AttributeError):
        return {}
    return {str(k): [float(x) for x in v] for k, v in parsed}


def dump_embeds(vectors: list[list[float]]) -> bytes:
    import json

    return json.dumps({"vectors": vectors}, separators=(",", ":")).encode()


def load_embeds(members: dict[str, bytes], path: str) -> list[list[float]]:
    """Per-reference embeddings out of a ``.char``, or empty when it predates them."""
    import json

    raw = members.get(str(path))
    if not raw:
        return []
    try:
        return [[float(x) for x in v] for v in (json.loads(raw).get("vectors") or [])]
    except (ValueError, TypeError, AttributeError):
        return []


def encoder_versions() -> list[dict[str, Any]]:
    """What the manifest records, so a centroid is never compared across encoder builds.

    A picked file folds into the version: the constants track the shipped builds, so on their own
    they would call centroids from someone else's encoder current, and cosine similarity across two
    encoders is a number with no meaning.
    """
    return [
        {"id": SFACE_ID, "version": _version(weights.SFACE_VERSION, "sface", weights.SFACE_FILE),
         "dim": 128},
        {"id": DINOV2_ID, "version": _version(weights.DINOV2_VERSION, "dinov2", weights.DINOV2_DIR),
         "dim": 768},
    ]


def _version(base: str, kind: str, default: str) -> str:
    name = chosen(kind, default)
    return base if name == default else f"{base}:{name}"


def encoder_versions_by_id() -> dict[str, str]:
    """The same table as ``encoder_versions``, shaped for a validity check."""
    return {str(e["id"]): str(e["version"]) for e in encoder_versions()}


def annotator_present(path: Path) -> bool:
    return path.exists()
