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

#: Square input YuNet and SFace expect after our own resize.
_DETECT_SIZE = 640

_models: dict[str, Any] = {}


def _yunet(width: int, height: int) -> Any:
    import cv2

    detector = _models.get("yunet")
    if detector is None:
        weights.ensure()
        detector = cv2.FaceDetectorYN.create(
            str(weights.yunet_path()), "", (width, height), FACE_CONFIDENCE
        )
        _models["yunet"] = detector
    detector.setInputSize((width, height))
    return detector


def _sface() -> Any:
    import cv2

    model = _models.get("sface")
    if model is None:
        weights.ensure()
        model = cv2.FaceRecognizerSF.create(str(weights.sface_path()), "")
        _models["sface"] = model
    return model


def _dinov2() -> tuple[Any, Any]:
    pair = _models.get("dinov2")
    if pair is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        weights.ensure()
        root = str(weights.dinov2_path())
        processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        model = AutoModel.from_pretrained(root, local_files_only=True, dtype=torch.float32)
        model.eval()
        pair = (processor, model)
        _models["dinov2"] = pair
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
) -> dict[str, Any] | None:
    """One blended score, or None when nothing was measurable - which is not a zero.

    ``face_refs`` are the per-reference face embeddings; without them the face term falls back to
    the centroid, which is what a character encoded before this existed carries.
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
    if subject_centroid:
        embedding = embed_subject(image)
        if embedding:
            subject_score = to_percent(cosine(embedding, subject_centroid))

    if face_score is not None and subject_score is not None:
        blended = FACE_WEIGHT * face_score + SUBJECT_WEIGHT * subject_score
    elif face_score is not None:
        blended = face_score
    elif subject_score is not None:
        blended = subject_score
    else:
        return None

    return {
        "score": round(blended, 1),
        "faceScore": face_score,
        "subjectScore": subject_score,
        "faceBearing": face_score is not None,
    }


def reference_agreement(face_refs: list[list[float]]) -> list[float]:
    """Each reference's mean SFace similarity to the others, 0-100."""
    if len(face_refs) < 2:
        return [100.0] * len(face_refs)
    out: list[float] = []
    for i, vector in enumerate(face_refs):
        others = [v for j, v in enumerate(face_refs) if j != i]
        out.append(round(sum(to_percent(cosine(vector, o)) for o in others) / len(others), 1))
    return out


def flagged_references(face_refs: list[list[float]]) -> list[int]:
    """Indices of references that may not be the same person. Face identity only - a different
    outfit or setting is exactly what the user was asked for and must not trip this."""
    if len(face_refs) < MIN_REFS_TO_FLAG:
        return []
    scores = reference_agreement(face_refs)
    return [i for i, s in enumerate(scores) if s < REFERENCE_AGREEMENT_FLOOR]


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
    """What the manifest records, so a centroid is never compared across encoder builds."""
    return [
        {"id": SFACE_ID, "version": weights.SFACE_VERSION, "dim": 128},
        {"id": DINOV2_ID, "version": weights.DINOV2_VERSION, "dim": 768},
    ]


def annotator_present(path: Path) -> bool:
    return path.exists()
