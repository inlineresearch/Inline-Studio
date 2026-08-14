"""Identity scoring: the blend, the fallback, and the centroid maths.

The vector maths is exact and runs everywhere. The encoder-backed assertions are gated on the
weights being fetchable, so the suite still runs offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.characters import charfile as cf
from inline_core.characters import scoring


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    assert scoring.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_ignores_magnitude() -> None:
    assert scoring.cosine([1.0, 0.0], [7.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_mismatched_or_empty_vectors_is_zero() -> None:
    assert scoring.cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    assert scoring.cosine([], [1.0]) == 0.0


def test_to_percent_clamps_a_negative_similarity_to_zero() -> None:
    """Anti-correlation is no match, not the opposite of one, so it must not land mid-scale."""
    assert scoring.to_percent(-0.8) == 0.0
    assert scoring.to_percent(1.0) == 100.0
    assert scoring.to_percent(0.5) == 50.0


def test_mean_vector_normalises_before_averaging() -> None:
    """Without this one high-magnitude reference would drag the centroid onto itself."""
    centroid = scoring.mean_vector([[100.0, 0.0], [0.0, 1.0]])
    assert centroid is not None
    assert centroid[0] == pytest.approx(centroid[1])


def test_mean_vector_of_nothing_is_none() -> None:
    assert scoring.mean_vector([]) is None
    assert scoring.mean_vector([[]]) is None


def test_score_blends_face_and_subject_with_the_declared_weights() -> None:
    face = [1.0, 0.0]
    subject = [1.0, 0.0]
    centroids = {scoring.SFACE_ID: face, scoring.DINOV2_ID: subject}

    calls: dict[str, object] = {}

    def fake_face(_image: object) -> list[float]:
        calls["face"] = True
        return [1.0, 0.0]

    def fake_subject(_image: object) -> list[float]:
        calls["subject"] = True
        return [0.0, 1.0]

    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face, scoring.embed_subject = fake_face, fake_subject  # type: ignore[assignment]
    try:
        result = scoring.score(object(), centroids)
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]

    assert result is not None
    assert result["faceScore"] == 100.0
    assert result["subjectScore"] == 0.0
    assert result["score"] == pytest.approx(scoring.FACE_WEIGHT * 100)
    assert result["faceBearing"] is True


def test_score_falls_back_to_the_subject_when_there_is_no_face() -> None:
    centroids = {scoring.SFACE_ID: [1.0, 0.0], scoring.DINOV2_ID: [1.0, 0.0]}

    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _image: None  # type: ignore[assignment]
    scoring.embed_subject = lambda _image: [1.0, 0.0]  # type: ignore[assignment]
    try:
        result = scoring.score(object(), centroids)
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]

    assert result is not None
    assert result["faceBearing"] is False
    assert result["faceScore"] is None
    assert result["score"] == 100.0


def test_score_returns_none_when_nothing_could_be_measured() -> None:
    """A missing measurement and a bad match are different facts, so this must not be a zero."""
    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _image: None  # type: ignore[assignment]
    scoring.embed_subject = lambda _image: None  # type: ignore[assignment]
    try:
        assert scoring.score(object(), {scoring.DINOV2_ID: [1.0]}) is None
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]
    assert scoring.score(object(), {}) is None


def test_centroids_round_trip_through_char_members() -> None:
    vector = scoring.normalise([0.3, 0.4, 0.5])
    members = {"scoring/centroid_sface.json": scoring.dump_centroid(vector, 3)}
    loaded = scoring.load_centroids(members, {scoring.SFACE_ID: "scoring/centroid_sface.json"})
    assert loaded[scoring.SFACE_ID] == pytest.approx(vector)


def test_a_corrupt_centroid_is_skipped_rather_than_crashing() -> None:
    members = {"scoring/bad.json": b"{not json"}
    assert scoring.load_centroids(members, {scoring.SFACE_ID: "scoring/bad.json"}) == {}
    assert scoring.load_centroids({}, {scoring.SFACE_ID: "scoring/missing.json"}) == {}


def test_a_moved_encoder_version_invalidates_a_stored_centroid() -> None:
    """Cosine similarity across two encoder builds is a number with no meaning."""
    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    manifest.scoring = {"encoders": scoring.encoder_versions()}
    assert cf.centroid_valid(manifest, scoring.SFACE_ID, scoring.weights.SFACE_VERSION)
    assert not cf.centroid_valid(manifest, scoring.SFACE_ID, "some-other-build")


@pytest.mark.skipif(
    not Path.home().joinpath(".cache/huggingface").is_dir(),
    reason="no HF cache for scoring weights",
)
def test_a_face_photo_scores_far_above_a_different_person() -> None:
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from inline_core.characters import encode

    ada = hf_hub_download("opencv/face_detection_yunet", "example_outputs/largest_selfie.jpg")
    someone_else = hf_hub_download("opencv/face_recognition_sface", "example_outputs/demo.jpg")

    doc = encode.char_encode([ada], name="Ada")
    centroids = scoring.load_centroids(doc.members, doc.manifest.scoring["centroids"])

    same = scoring.score(Image.open(ada), centroids)
    other = scoring.score(Image.open(someone_else), centroids)
    assert same is not None and other is not None
    assert same["score"] > 95
    assert other["score"] < 40


def test_adding_references_never_lowers_the_face_score() -> None:
    """The centroid made every added reference lower the score, because views of one person sit
    far apart in SFace space and their average matches none of them."""
    probe = [1.0, 0.0]
    near, far = [1.0, 0.0], [0.0, 1.0]

    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _i: probe  # type: ignore[assignment]
    scoring.embed_subject = lambda _i: None  # type: ignore[assignment]
    try:
        one = scoring.score(object(), {scoring.SFACE_ID: near}, [near])
        many = scoring.score(object(), {scoring.SFACE_ID: near}, [near, far, far])
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]

    assert one is not None and many is not None
    assert one["faceScore"] == 100.0
    assert many["faceScore"] == 100.0, "a matching view must still score 100 beside unlike ones"


def test_a_character_encoded_before_per_reference_embeds_still_scores() -> None:
    """Old files carry only a centroid, so the face term falls back to it."""
    original = (scoring.embed_face, scoring.embed_subject)
    scoring.embed_face = lambda _i: [1.0, 0.0]  # type: ignore[assignment]
    scoring.embed_subject = lambda _i: None  # type: ignore[assignment]
    try:
        result = scoring.score(object(), {scoring.SFACE_ID: [1.0, 0.0]}, None)
    finally:
        scoring.embed_face, scoring.embed_subject = original  # type: ignore[assignment]
    assert result is not None and result["faceScore"] == 100.0


def test_per_reference_embeds_round_trip() -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    members = {"scoring/embeds_sface.json": scoring.dump_embeds(vectors)}
    assert scoring.load_embeds(members, "scoring/embeds_sface.json") == vectors
    assert scoring.load_embeds({}, "scoring/embeds_sface.json") == []
    assert scoring.load_embeds({"scoring/x.json": b"{bad"}, "scoring/x.json") == []


# --- reference outlier detection -------------------------------------------------------------


#: Wide enough that a cluster can perturb its own dimensions without drifting toward the impostor,
#: which is what real 128-d SFace space does and a 2-d toy cannot.
_DIMS = 8


def _same_person(n: int) -> list[list[float]]:
    """n vectors clustered on one axis, the way views of one person cluster."""
    out = []
    for i in range(n):
        v = [0.0] * _DIMS
        v[0] = 1.0
        v[2 + (i % (_DIMS - 2))] = 0.15
        out.append(scoring.normalise(v))
    return out


def _different_person() -> list[float]:
    v = [0.0] * _DIMS
    v[1] = 1.0
    return scoring.normalise(v)


def test_a_consistent_reference_set_flags_nothing() -> None:
    assert scoring.flagged_references(_same_person(4)) == []


def test_one_different_person_flags_exactly_that_reference() -> None:
    """Best-match means a single wrong reference is a backdoor, so it has to be caught at encode."""
    refs = _same_person(3) + [_different_person()]
    assert scoring.flagged_references(refs) == [3]


def test_agreement_is_reported_per_reference() -> None:
    refs = _same_person(3) + [_different_person()]
    scores = scoring.reference_agreement(refs)
    assert len(scores) == 4
    assert scores[3] < scoring.REFERENCE_AGREEMENT_FLOOR
    assert all(s > scoring.REFERENCE_AGREEMENT_FLOOR for s in scores[:3])


def test_two_references_are_never_flagged() -> None:
    """With two, "agreement with the others" is one pairwise number and cannot say which is odd."""
    refs = _same_person(1) + [_different_person()]
    assert scoring.flagged_references(refs) == []


def test_a_single_reference_agrees_with_itself() -> None:
    assert scoring.reference_agreement([[1.0, 0.0]]) == [100.0]
    assert scoring.flagged_references([[1.0, 0.0]]) == []


# --- lookalike discrimination ------------------------------------------------------------------


def test_a_lookalike_scores_far_below_the_real_person() -> None:
    """Best-match only needs one close view to fire, so the failure mode that matters is a
    near-duplicate rather than an obviously different face. Real SFace embeddings from LFW: the
    nearest non-matching identity to this character out of 260 candidates."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "reference" / "lookalike_faces.json"
    data = json.loads(fixture.read_text())
    refs = data["refs"]

    def face_score(probe: list[float]) -> float:
        return max(scoring.to_percent(scoring.cosine(probe, r)) for r in refs)

    same = face_score(data["same_person"])
    lookalike = face_score(data["lookalike_face"])
    assert same > 70, f"a held-out view of {data['target']} should score high, got {same}"
    assert lookalike < 50, f"{data['lookalike']} must not pass as {data['target']}: {lookalike}"
    # Measured 80.3 vs 37.4; the margin is the thing worth guarding, not either number alone.
    assert same - lookalike > 25
