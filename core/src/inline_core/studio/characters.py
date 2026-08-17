"""Studio-side character library: the RPC surface over ``characters/`` plus take scoring.

Encoding takes seconds, so every entry point pins its project with ``project_ref``/``bind`` - the
open project may have changed by the time it returns. CPU-only, so it can run during a render.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..characters import charfile as cf
from ..characters import encode, library, scoring
from ..config import models_dir

logger = logging.getLogger("inline_core.studio.characters")

CHANGED_EVENT = "events:charactersChanged"
PROGRESS_EVENT = "events:characterProgress"

#: The one config the gate proved, per architecture. Not a menu: a second "fast" tier measured no
#: better, and the scope must stay "full" - narrowing it bound nothing on either arch.
BUILD_STEPS = 600
#: Every value here has a measured score behind it. 1200 peaked on faces (80.0 against 68.0 at
#: 600); 2000 fell back to 72.9, so nothing above it is offered rather than guessed at.
BUILD_STEP_CHOICES = (600, 1200, 2000)
_BUILD_CONFIG: dict[str, dict[str, Any]] = {
    "flux2": {"arch": "flux2", "baseMode": ""},
    # Turbo plus adapter is the path for anyone holding only Turbo, and "auto" because Krea 2's
    # base is 26GB: hardcoding "none" is what OOMed the first attempt.
    "krea2": {"arch": "krea2", "baseMode": "turbo_adapter"},
}
_BUILD_COMMON: dict[str, Any] = {
    "rank": 16, "alpha": 16, "resolution": 512, "batchSize": 1, "learningRate": 1e-4,
    "saveEvery": 100000, "saveSnapshots": False, "captionDropout": 0.0, "flipAugment": False,
    "offload": False, "baseQuant": "auto", "loraScope": "full", "trainingMode": "lora",
    "clipSeconds": 0, "clipWindow": 0, "steps": BUILD_STEPS,
}


class Characters:
    """The `characters:*` channels, backed by ``models/characters/``."""

    def __init__(
        self, store: Any, events: Any, on_change: Any = None, training: Any = None
    ) -> None:
        self._store = store
        self._events = events
        # The catalog caches its scan, so a new character stays invisible to the node dropdown
        # until something rescans. Same hook the trained-LoRA path uses.
        self._on_change = on_change
        self._training = training
        #: run id -> (character file, arch, steps), so a finished run knows what it belongs to.
        self._builds: dict[str, tuple[str, str, int]] = {}
        if training is not None:
            training.add_done_listener(self._on_training_done)

    # --- reads ----------------------------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        return sorted(
            library.summaries(), key=lambda row: int(row.get("modifiedAt") or 0), reverse=True
        )

    def get(self, file: str) -> dict[str, Any]:
        path = self._require(file)
        doc = cf.read(path)
        if self._scoring_stale(doc.manifest):
            # Refs are truth and scoring is cache, so an outdated character rebuilds itself.
            logger.info("Rebuilding stale scoring for %s", path.name)
            self._recompile(path, self._ref_files(doc), doc)
            doc = cf.read(path)
        summary = self._summary(path, doc)
        summary["refUrls"] = [
            f"/character-ref/{path.name}/{index}" for index in range(len(doc.manifest.refs))
        ]
        summary["faceBearing"] = bool(doc.manifest.scoring.get("face_bearing"))
        summary.update(encode.flags_for(doc))
        return summary

    # --- writes ---------------------------------------------------------------------------------

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str((payload or {}).get("name") or "").strip()
        asset_ids = list((payload or {}).get("assetIds") or [])
        description = str((payload or {}).get("description") or "")
        if not name:
            raise ValueError("A character needs a name.")
        if not asset_ids:
            raise ValueError("A character needs at least one reference image.")

        ref = self._store.project_ref()
        with self._store.bind(ref) as conn:
            paths = self._asset_paths(conn, ref.folder, asset_ids)
        doc = encode.char_encode(
            paths, name=name, description=description, on_progress=self._progress(name)
        )
        path = library.save(doc)
        self._changed()
        return self._summary(path, doc)

    def create_from_take(self, take_id: str, name: str) -> dict[str, Any]:
        """Turn a generated take into a character, which is how a good render becomes reusable."""
        ref = self._store.project_ref()
        with self._store.bind(ref) as conn:
            path = self._take_path(conn, ref.folder, take_id)
        doc = encode.char_encode([path], name=name, on_progress=self._progress(name))
        saved = library.save(doc)
        self._changed()
        return self._summary(saved, doc)

    def rename(self, file: str, name: str) -> dict[str, Any]:
        """Rename in place - the filename is what every node already picking it stores."""
        cleaned = str(name or "").strip()
        if not cleaned:
            raise ValueError("A character needs a name.")
        return self._edit(file, lambda doc: setattr(doc.manifest, "name", cleaned))

    def set_description(self, file: str, description: str) -> dict[str, Any]:
        def apply(doc: cf.CharDoc) -> None:
            member = str(doc.manifest.text.get("path") or "text/description.md")
            data = str(description or "").encode("utf-8")
            doc.members[member] = data
            doc.manifest.text = {"path": member, "sha256": cf.sha256_bytes(data)}

        # Refs are untouched, so the payload fingerprint still matches and nothing recompiles.
        return self._edit(file, apply)

    def add_refs(self, file: str, asset_ids: list[str]) -> dict[str, Any]:
        """Add references immediately; rebuilding is explicit, so no edit waits on an encoder."""
        if not asset_ids:
            raise ValueError("Pick at least one image to add.")
        path = self._require(file)
        doc = cf.read(path)
        ref = self._store.project_ref()
        with self._store.bind(ref) as conn:
            paths = self._asset_paths(conn, ref.folder, asset_ids)
        encode.append_refs(doc, paths)
        cf.write(path, doc)
        self._changed()
        return self._summary(path, doc)

    def remove_ref(self, file: str, index: int) -> dict[str, Any]:
        path = self._require(file)
        doc = cf.read(path)
        encode.drop_ref(doc, index)
        cf.write(path, doc)
        self._changed()
        return self._summary(path, doc)

    def set_apply_mode(self, file: str, arch: str, mode: str) -> dict[str, Any]:
        """Choose how this character applies for one model: its references, or its adapter."""
        if mode not in ("reference", "lora"):
            raise ValueError("Mode must be 'reference' or 'lora'.")
        return self._edit(file, lambda doc: doc.manifest.apply.__setitem__(arch, mode))

    def rebuild(self, file: str) -> dict[str, Any]:
        """Recompile scoring and the reference payload from whatever the references are now."""
        path = self._require(file)
        doc = cf.read(path)
        return self._recompile(path, self._ref_files(doc), doc)

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

            with Image.open(image_path) as handle:
                return scoring.score(
                    handle.convert("RGB"), centroids, face_refs, subject_refs, framings
                )
        except Exception as error:  # noqa: BLE001 - scoring is never worth failing a render over
            logger.warning("Could not score %s against %s: %s", image_path, chosen, error)
            return None

    # --- internals ------------------------------------------------------------------------------

    def _require(self, file: str) -> Path:
        path = library.resolve(file)
        if path is None:
            raise ValueError(f"Character {file!r} is no longer in models/characters/.")
        return path

    def _edit(self, file: str, apply: Any) -> dict[str, Any]:
        path = self._require(file)
        doc = cf.read(path)
        apply(doc)
        doc.manifest.modified_at = int(time.time())
        cf.write(path, doc)
        self._changed()
        return self._summary(path, doc)

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
        return bool(manifest.refs) and not manifest.scoring.get("refFramings")

    def _recompile(self, path: Path, refs: list[Path], previous: cf.CharDoc) -> dict[str, Any]:
        """Rebuild from a new reference set, keeping char_id and filename so nothing unpicks."""
        doc = encode.char_encode(
            refs,
            name=previous.manifest.name,
            description=self._description(previous),
            char_id=previous.manifest.char_id,
            created_at=previous.manifest.created_at,
            on_progress=self._progress(previous.manifest.name),
        )
        cf.write(path, doc)
        self._changed()
        return self._summary(path, doc)

    def _ref_files(self, doc: cf.CharDoc) -> list[Path]:
        """The character's own refs written out, so a rebuild reads truth not the user's library."""
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="inline-char-refs-"))
        out: list[Path] = []
        for index, ref in enumerate(doc.manifest.refs):
            data = doc.members.get(str(ref.get("path")))
            if data is None:
                continue
            target = root / f"{index:03d}.png"
            target.write_bytes(data)
            out.append(target)
        return out

    def _description(self, doc: cf.CharDoc) -> str:
        raw = doc.members.get(str(doc.manifest.text.get("path") or ""))
        return raw.decode("utf-8", errors="replace") if raw else ""

    def _summary(self, path: Path, doc: cf.CharDoc) -> dict[str, Any]:
        manifest = doc.manifest
        return {
            "file": path.name,
            "charId": manifest.char_id,
            "name": manifest.name or path.stem,
            "refs": len(manifest.refs),
            "createdAt": manifest.created_at,
            "modifiedAt": manifest.modified_at,
            "description": self._description(doc),
            "hints": encode.hints_for(manifest),
            "sizeBytes": path.stat().st_size if path.is_file() else 0,
            "builds": self._builds_for(manifest),
            "needsRebuild": encode.needs_rebuild(manifest),
        }

    def _base_ready(self, arch: str) -> bool:
        """Whether the checkpoint this arch trains against is on disk, checked here rather than
        twenty minutes into a run: the trainer only finds out after precaching."""
        config = _BUILD_CONFIG.get(arch)
        if config is None:
            return False
        try:
            from ..config import models_dir as root
            from ..training import models as training_models

            return bool(training_models._base_file(root(), arch, str(config["baseMode"])))
        except Exception:
            return False

    def _builds_for(self, manifest: cf.Manifest) -> list[dict[str, Any]]:
        """What each architecture can apply today, so the UI never offers a build that is already
        current or claims a stale adapter is ready."""
        rows: list[dict[str, Any]] = []
        for arch, label in (("flux2", "FLUX.2"), ("krea2", "Krea 2")):
            key = encode.payload_key(arch, encode.PAYLOAD_LORA)
            trained = encode.lora_payload(manifest, arch) is not None
            valid = trained and cf.payload_valid(manifest, key, encode.LORA_PAYLOAD_VERSION)
            rows.append({
                "arch": arch,
                "label": label,
                # Only FLUX.2 has a reference channel; on Krea 2 the adapter is the only route.
                "reference": arch == "flux2",
                "lora": "ready" if valid else "stale" if trained else "none",
                # What a render would actually use right now, resolved the same way apply does.
                "mode": manifest.apply.get(arch) or ("lora" if valid else "reference"),
                "baseReady": self._base_ready(arch),
            })
        return rows

    def _asset_paths(
        self, conn: sqlite3.Connection, folder: Path, asset_ids: list[str]
    ) -> list[Path]:
        found: dict[str, Path] = {}
        for asset_id in asset_ids:
            row = conn.execute(
                "SELECT file_path, kind FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                raise ValueError("One of those images is no longer in the library.")
            if str(row[1]) != "image":
                raise ValueError("A character's references have to be images.")
            found[asset_id] = folder / str(row[0])
        # Preserve the caller's order: it is the order the references will be numbered in.
        return [found[asset_id] for asset_id in asset_ids]

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

    async def build(
        self, file: str, arch: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Train this character's adapter for ``arch``, returning the run it queued."""
        if self._training is None:
            raise RuntimeError("Training is not available on this install.")
        config = _BUILD_CONFIG.get(arch)
        if config is None:
            raise ValueError(f"No character build is defined for {arch}.")
        opts = options or {}
        steps = int(opts.get("steps") or BUILD_STEPS)
        if steps not in BUILD_STEP_CHOICES:
            raise ValueError(f"Steps must be one of {', '.join(map(str, BUILD_STEP_CHOICES))}.")
        path = self._require(file)
        doc = cf.read(path)
        description = self._description(doc).strip()
        if not description:
            raise ValueError(
                "Add a description first. The adapter binds to it, so without one there is "
                "nothing for a prompt to summon."
            )
        if not self._base_ready(arch):
            raise ValueError(
                f"The {arch} base checkpoint is not downloaded yet. Add it from a node's model "
                "popup first, then train."
            )

        name = doc.manifest.name
        report = self._progress(name)
        report(0.1, "Preparing the training set…")
        dataset = self._training.create_dataset({"name": f"{name} ({arch})"})
        folder = self._dataset_dir(doc)
        staged = self._training.stage_from_path(str(folder))
        items = self._training.commit_staged(dataset["id"], staged)

        if opts.get("autoCaption"):
            # The captioner loads a model, so this is the one phase that runs before the queue and
            # can take minutes on its own.
            report(0.4, f"Captioning {len(items)} references…")
            await self._training.auto_caption(dataset["id"], True, opts.get("captioner") or None)
        else:
            for item in items:
                self._training.set_caption(item["id"], description)

        report(1.0, "Queued")
        run = self._training.start(dataset["id"], {**_BUILD_COMMON, **config, "steps": steps})
        self._builds[str(run["id"])] = (path.name, arch, steps)
        return run

    def _dataset_dir(self, doc: cf.CharDoc) -> Path:
        """The character's own refs written out, since the source assets may be long gone."""
        import tempfile

        folder = Path(tempfile.mkdtemp(prefix="char-build-"))
        for index, ref in enumerate(doc.manifest.refs):
            data = doc.members.get(str(ref.get("path") or ""))
            if data:
                (folder / f"{index:04d}.png").write_bytes(data)
        return folder

    def _on_training_done(self, run_id: str, output_rel: str) -> None:
        """Claim a finished run's adapter as this character's payload for its arch."""
        claim = self._builds.pop(str(run_id), None)
        if claim is None:
            return
        file, arch, steps = claim
        path = library.resolve(file)
        if path is None:
            logger.warning("Character %s went away before its %s adapter landed", file, arch)
            return
        adapter = models_dir() / output_rel
        if not adapter.is_file():
            logger.warning("Trained adapter missing for %s: %s", file, output_rel)
            return
        doc = cf.read(path)
        encode.set_lora_payload(
            doc.manifest,
            doc.members,
            adapter.read_bytes(),
            arch=arch,
            base=adapter.name,
            rank=int(_BUILD_COMMON["rank"]),
            steps=steps,
            resolution=int(_BUILD_COMMON["resolution"]),
        )
        cf.write(path, doc)
        logger.info("Wrote the %s adapter into %s", arch, path.name)
        self._changed()

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
