"""Studio-side character library: the RPC surface over ``characters/`` plus take scoring.

Encoding takes seconds, so every entry point pins its project with ``project_ref``/``bind`` - the
open project may have changed by the time it returns. CPU-only, so it can run during a render.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from ..characters import charfile as cf
from ..characters import encode, library, scoring, weights

logger = logging.getLogger("inline_core.studio.characters")

#: Raised so the client can open the model popup instead of the encoders downloading silently.
ENCODERS_MISSING = (
    "The character encoders are not downloaded yet: face_detection_yunet_2023mar.onnx, "
    "face_recognition_sface_2021dec.onnx and dinov2-base, about 385MB in models/annotators."
)

CHANGED_EVENT = "events:charactersChanged"
PROGRESS_EVENT = "events:characterProgress"

#: Takes whose score has to come from frames rather than from the file itself.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}

#: How many frames a video take is measured on. Each one costs a full SFace + DINOv2 pass, and this
#: runs inline while the take is saved, so it buys robustness rather than precision.
SCORE_FRAMES = 5

#: Skipped at each end, where a video is least settled and a low score would say more about the
#: first moments than about the character.
_EDGE_SECONDS = 0.5


def _sample_frames(src: Path, count: int = SCORE_FRAMES) -> list[Any]:
    """Evenly spaced frames as PIL images, or [] when ffmpeg cannot read the file."""
    import io
    import subprocess

    from PIL import Image

    from ..ffmpeg import ffmpeg_exe

    exe = ffmpeg_exe()
    if exe is None or not src.is_file():
        return []
    frames: list[Any] = []
    duration = _duration_seconds(src)
    if duration is None or duration <= 0:
        return []
    span = max(duration - 2 * _EDGE_SECONDS, 0.0)
    # One decode per frame: seeking is cheaper than decoding the whole clip for five stills.
    for index in range(count):
        offset = _EDGE_SECONDS + (span * (index + 0.5) / count if span else 0.0)
        try:
            proc = subprocess.run(
                [exe, "-v", "quiet", "-ss", f"{offset:.3f}", "-i", str(src),
                 "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if not proc.stdout:
            continue
        try:
            with Image.open(io.BytesIO(proc.stdout)) as handle:
                frames.append(handle.convert("RGB"))
        except Exception:  # noqa: BLE001 - a frame that will not decode is one fewer sample
            continue
    return frames


def _duration_seconds(src: Path) -> float | None:
    """The clip's length via ffprobe, or None. ffprobe is often absent, so this is best-effort."""
    import subprocess

    from ..ffmpeg import ffprobe_exe

    exe = ffprobe_exe()
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(src)],
            capture_output=True,
            timeout=30,
        )
        return float(proc.stdout.decode().strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _score_video(
    src: Path,
    centroids: dict[str, list[float]],
    face_refs: list[list[float]],
    subject_refs: list[list[float]],
    framings: list[float],
) -> dict[str, Any] | None:
    """One score for a clip: the median across the frames that measured, never a mean.

    A frame where no face was found returns None from ``score`` and drops out rather than counting
    as a zero, and the median survives one blurred frame and one lucky one alike. ``frames`` rides
    along so a number from two samples is not read as a number from five.
    """
    from statistics import median

    measured = [
        result
        for frame in _sample_frames(src)
        if (result := scoring.score(frame, centroids, face_refs, subject_refs, framings))
    ]
    if not measured:
        return None
    out = dict(measured[len(measured) // 2])
    out["score"] = round(median(float(m["score"]) for m in measured), 1)
    # Face-only if any sampled frame could not be spoken to by the subject term.
    out["subjectCounted"] = all(m.get("subjectCounted", True) for m in measured)
    out["frames"] = len(measured)
    return out


class Characters:
    """The `characters:*` channels, backed by ``models/characters/``."""

    def __init__(self, store: Any, events: Any, on_change: Any = None) -> None:
        self._store = store
        self._events = events
        # The catalog caches its scan, so a new character stays invisible to the node dropdown
        # until something rescans. Same hook the trained-LoRA path uses.
        self._on_change = on_change

    # --- reads ----------------------------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        return sorted(
            library.summaries(), key=lambda row: int(row.get("modifiedAt") or 0), reverse=True
        )


    # --- writes ---------------------------------------------------------------------------------


    def create_from_take(self, take_id: str, name: str) -> dict[str, Any]:
        """Turn a generated take into a character, which is how a good render becomes reusable."""
        self._require_encoders()
        ref = self._store.project_ref()
        with self._store.bind(ref) as conn:
            path = self._take_path(conn, ref.folder, take_id)
        doc = encode.char_encode([path], name=name, on_progress=self._progress(name))
        saved = library.save(doc)
        self._changed()
        return self._summary(saved, doc)







    def delete(self, file: str) -> bool:
        removed = library.delete(file)
        if removed:
            self._changed()
        return removed

    # --- scoring --------------------------------------------------------------------------------

    def score_take(self, image_path: Path | str, chosen: str) -> dict[str, Any] | None:
        """Continuity score, or None. Never raises: metadata must not fail a render."""
        try:
            path = library.resolve(chosen)
            if path is None:
                return None
            doc = cf.read(path)
            # Before anything reads a version: a node's encoder pick must not decide this.
            scoring.use_encoders_from(doc.manifest.scoring)
            # Refs are truth and scoring is cache, so a character written before the current
            # encoders is rebuilt here rather than scoring against a centroid nothing can compare.
            if self._scoring_stale(doc.manifest):
                doc = self._rescore(path, doc)
            centroids = scoring.load_centroids(
                doc.members, doc.manifest.scoring.get("centroids") or {}
            )
            # Similarity across two encoder builds is meaningless, so a stale term is dropped.
            for encoder_id, version in scoring.encoder_versions_by_id().items():
                if encoder_id in centroids and not cf.centroid_valid(
                    doc.manifest, encoder_id, version
                ):
                    centroids.pop(encoder_id, None)
            if not centroids:
                return None
            face_refs = scoring.load_embeds(
                doc.members, str(doc.manifest.scoring.get("faceEmbeds") or "")
            )
            subject_refs = scoring.load_embeds(
                doc.members, str(doc.manifest.scoring.get("subjectEmbeds") or "")
            )
            framings = [float(f) for f in (doc.manifest.scoring.get("refFramings") or [])]
            from PIL import Image

            path_in = Path(image_path)
            if path_in.suffix.lower() in _VIDEO_SUFFIXES:
                return _score_video(path_in, centroids, face_refs, subject_refs, framings)
            with Image.open(path_in) as handle:
                return scoring.score(
                    handle.convert("RGB"), centroids, face_refs, subject_refs, framings
                )
        except Exception as error:  # noqa: BLE001 - scoring is never worth failing a render over
            logger.warning("Could not score %s against %s: %s", image_path, chosen, error)
            return None

    # --- internals ------------------------------------------------------------------------------

    def _scoring_stale(self, manifest: cf.Manifest) -> bool:
        """Whether this character's scoring predates what the current encoders record.

        Two ways to be stale: an encoder version has moved, so its centroid cannot be compared; or
        the manifest lacks the per-reference framings the subject term needs to know whether it can
        speak to a take at all.
        """
        for encoder_id, version in scoring.encoder_versions_by_id().items():
            if manifest.scoring.get("centroids", {}).get(encoder_id) and not cf.centroid_valid(
                manifest, encoder_id, version
            ):
                return True
        # A reference was added or dropped since, so every stored per-reference list is misaligned.
        recorded = manifest.scoring.get("refCount")
        if recorded is not None and int(recorded) != len(manifest.refs):
            return True
        # Key presence, not truthiness: a character whose references carry no detectable face has
        # an honestly empty list, and reading that as stale rescored it on every take forever.
        return bool(manifest.refs) and "refFramings" not in manifest.scoring

    def _rescore(self, path: Path, previous: cf.CharDoc) -> cf.CharDoc:
        """Recompute scoring from the character's own refs, in place.

        Never through `char_encode`: that builds a fresh manifest, so the write below would put the
        loss of the trained adapter and every non-flux payload on disk.
        """
        encode.rescore(previous, self._progress(previous.manifest.name))
        cf.write(path, previous)
        self._changed()
        return previous

    def _require(self, file: str) -> Path:
        path = library.resolve(file)
        if path is None:
            raise ValueError(f"Character {file!r} is no longer in models/characters/.")
        return path





    def _description(self, doc: cf.CharDoc) -> str:
        raw = doc.members.get(str(doc.manifest.text.get("path") or ""))
        return raw.decode("utf-8", errors="replace") if raw else ""

    def _summary(self, path: Path, doc: cf.CharDoc) -> dict[str, Any]:
        manifest = doc.manifest
        return {
            "file": path.name,
            "charId": manifest.char_id,
            "name": manifest.name or path.stem,
            "refs": len(encode.originals(manifest)),
            "harvested": len(encode.harvested(manifest)),
            "flagged": list(manifest.scoring.get("flaggedRefs") or []),
            "createdAt": manifest.created_at,
            "modifiedAt": manifest.modified_at,
            "description": self._description(doc),
            "hints": encode.hints_for(manifest),
            "sizeBytes": path.stat().st_size if path.is_file() else 0,
            "needsRebuild": encode.needs_rebuild(manifest),
        }




    def _take_path(self, conn: sqlite3.Connection, folder: Path, take_id: str) -> Path:
        row = conn.execute("SELECT file_path, kind FROM takes WHERE id = ?", (take_id,)).fetchone()
        if row is None:
            # A Core node's takes live in the moodboard item's JSON rather than the takes table.
            return self._core_take_path(conn, folder, take_id)
        if str(row[1]) != "image":
            raise ValueError("Only an image take can become a character.")
        return folder / str(row[0])

    def _core_take_path(self, conn: sqlite3.Connection, folder: Path, take_id: str) -> Path:
        import json

        for (data,) in conn.execute("SELECT data FROM moodboard_items WHERE data IS NOT NULL"):
            try:
                core = (json.loads(data) or {}).get("core") or {}
            except (ValueError, TypeError):
                continue
            for entry in core.get("outputs") or []:
                if entry.get("takeId") == take_id and entry.get("kind") == "image":
                    return folder / str(entry["filePath"])
        raise ValueError("That take is no longer available.")

    # --- building a LoRA payload ----------------------------------------------------------------




    def _require_encoders(self) -> None:
        """Encoding pulls ~385MB the first time; the popup asks before it happens."""
        if not weights.present():
            raise ValueError(ENCODERS_MISSING)

    def _progress(self, name: str) -> encode.Progress:
        """Stream an encode's phases; they arrive while the RPC call is still in flight."""

        def report(fraction: float, status: str) -> None:
            self._events.broadcast(
                PROGRESS_EVENT, {"name": name, "fraction": fraction, "status": status}
            )

        return report

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()
        self._events.broadcast(CHANGED_EVENT, {})
