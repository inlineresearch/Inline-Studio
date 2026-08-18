"""Character creation as graph nodes: identity once, then one payload per model.

A `.char` is identity (references, description, SFace and DINOv2 embeddings) plus payloads keyed
per model. Identity is model-independent, so adding a fifth model never re-encodes it - which is
why these are separate nodes rather than one. Payload nodes stay pure functions of (identity,
model, config); only `character/write` touches the file, so a fan-out cannot race.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...characters import charfile as cf
from ...characters import encode, library
from ...graph.descriptor import NodeDescriptor, Option, ParamField, Port, Widget
from ...graph.runners import NodeResult, NodeRunner
from ...graph.schema import Node, PortKind
from ...runtime.context import ExecutionContext
from ...runtime.progress import Phase
from ...takes import AssetRef
from ..pipeline_runtime import progress_event

logger = logging.getLogger("inline_core.character")


@dataclass
class Identity:
    """A character in flight. ``file`` is set once it exists on disk, which is what generation
    needs: applying one resolves payloads through the extraction cache, and that is keyed by the
    file's content. An identity straight out of Encode has none until it is written."""

    doc: cf.CharDoc
    file: str = ""

    @property
    def name(self) -> str:
        return self.doc.manifest.name


@dataclass
class Payload:
    """One model's compiled artefact, applied to the doc when the character is written."""

    arch: str
    kind: str
    apply: Any


ENCODE = NodeDescriptor(
    type="character/encode",
    title="Encode Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("images", "References", PortKind.IMAGE_LIST, required=True),
        Port("description", "Description", PortKind.TEXT, required=False),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        ParamField("name", "Name", Widget.TEXT, ""),
        ParamField("description", "Description", Widget.TEXTAREA, ""),
    ),
)

COMPILE_REFS = NodeDescriptor(
    type="character/references",
    title="Compile References",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(Port("character", "Character", PortKind.CHARACTER, required=True),),
    outputs=(Port("payload", "Payload", PortKind.PAYLOAD),),
    params=(
        # Only models with a reference channel; the rest take a character as a trained adapter.
        ParamField(
            "arch", "Model", Widget.SELECT, encode.FLUX2_KLEIN_ARCH,
            options=tuple(Option(value=a, label=a) for a in encode.REFERENCE_POLICIES),
        ),
    ),
)

LOAD = NodeDescriptor(
    type="character/load",
    title="Load Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(ParamField("file", "Character", Widget.SELECT, "", options_from="characters"),),
)

DATASET = NodeDescriptor(
    type="character/dataset",
    title="Character to Dataset",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(Port("character", "Character", PortKind.CHARACTER, required=True),),
    outputs=(
        Port("images", "References", PortKind.IMAGE_LIST),
        Port("captions", "Captions", PortKind.TEXT),
    ),
    params=(),
)

ATTACH = NodeDescriptor(
    type="character/adapter",
    title="Attach Adapter",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("lora", "LoRA", PortKind.LORA, required=True),
    ),
    outputs=(Port("payload", "Payload", PortKind.PAYLOAD),),
    params=(
        # Read from the adapter's own metadata when it has any; these are the fallback for one
        # trained before provenance was written.
        ParamField("arch", "Model", Widget.TEXT, ""),
        ParamField("base", "Trained against", Widget.TEXT, ""),
    ),
)

EDIT = NodeDescriptor(
    type="character/edit",
    title="Edit Character",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("images", "Add references", PortKind.IMAGE_LIST, required=False),
        Port("description", "Description", PortKind.TEXT, required=False),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        ParamField("name", "Rename to", Widget.TEXT, ""),
        ParamField("description", "Description", Widget.TEXTAREA, ""),
        ParamField("drop", "Remove references", Widget.TEXT, ""),
    ),
)

WRITE = NodeDescriptor(
    type="character/write",
    title="Write .char",
    category="Character",
    icon="sparkles",
    output_kind=None,
    inputs=(
        Port("character", "Character", PortKind.CHARACTER, required=True),
        Port("payloads", "Payloads", PortKind.PAYLOAD_LIST, required=False),
    ),
    outputs=(Port("character", "Character", PortKind.CHARACTER),),
    params=(
        # On the node face: where the character lands is worth seeing without opening Adjust.
        ParamField("filename", "Save as (name only)", Widget.TEXT, "", on_face=True),
        # Only bites when a model has both, which is the one case the file cannot decide alone.
        ParamField(
            "apply", "When a model has both", Widget.SELECT, "auto",
            options=(
                Option(value="auto", label="Auto (the adapter)"),
                Option(value="reference", label="Its references"),
                Option(value="lora", label="Its adapter"),
            ),
        ),
    ),
)


#: Told the saved filename when Write lands one, so the catalog and the library list catch up.
_on_saved: list[Any] = []


def set_save_listener(callback: Any) -> None:
    """A character written by a graph must reach the Load Character picker without a restart."""
    _on_saved.clear()
    _on_saved.append(callback)


def register_character_nodes(registry: Any) -> None:
    """Register the character graph nodes. Called best-effort by server.bootstrap."""
    registry.register(ENCODE, EncodeCharacterRunner())
    registry.register(LOAD, LoadCharacterRunner())
    registry.register(DATASET, CharacterDatasetRunner())
    registry.register(EDIT, EditCharacterRunner())
    registry.register(COMPILE_REFS, CompileReferencesRunner())
    registry.register(ATTACH, AttachAdapterRunner())
    registry.register(WRITE, WriteCharacterRunner())


def _first(values: list[Any] | None) -> Any:
    return values[0] if values else None


class EncodeCharacterRunner(NodeRunner):
    """References plus a description into an identity: crops, embeddings, framings, hints."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        from ...characters import weights

        if not weights.present():
            raise ValueError(
                "The character encoders are not downloaded yet: "
                "face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx and "
                "dinov2-base, about 385MB in models/annotators."
            )
        refs = list(inputs.get("images") or [])
        if not refs:
            raise ValueError("A character needs at least one reference image.")
        name = str(node.params.get("name") or "").strip() or "Character"
        # A wired description wins over the typed one, so a Prompt node can drive it.
        description = str(_first(inputs.get("description")) or node.params.get("description") or "")

        paths = [_image_path(ref) for ref in refs]

        def report(fraction: float, status: str) -> None:
            ctx.emitter.emit(progress_event(ctx, node, Phase.ENCODE, fraction, status=status))

        doc = encode.char_encode(paths, name=name, description=description, on_progress=report)
        logger.info("Encoded character %s from %d reference(s)", name, len(paths))
        return NodeResult(outputs={"character": Identity(doc=doc)})


class EditCharacterRunner(NodeRunner):
    """Change a character without re-encoding it: add or drop references, rename, re-describe.

    Every edit is cheap because none of it touches an encoder; scoring and payloads simply go stale,
    and the fingerprint check already treats a changed reference set as invalid.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Edit Character needs a character.")
        # Copied, because the upstream node's output is cached and every other reader shares it.
        doc = copy.deepcopy(identity.doc)

        # Dropped first, and from the back, so the numbers mean the strip the user is looking at.
        for index in sorted(_positions(node.params.get("drop")), reverse=True):
            encode.drop_ref(doc, index)
        added = [_image_path(ref) for ref in inputs.get("images") or []]
        if added:
            encode.append_refs(doc, added)

        name = str(node.params.get("name") or "").strip()
        if name:
            doc.manifest.name = name
        # A wired description wins over the typed one, so a Prompt node can drive it.
        description = str(_first(inputs.get("description")) or node.params.get("description") or "")
        if description:
            _set_description(doc, description)
        logger.info(
            "Edited %s: +%d ref(s), -%d ref(s)", doc.manifest.name, len(added),
            len(_positions(node.params.get("drop"))),
        )
        return NodeResult(outputs={"character": Identity(doc=doc, file=identity.file)})


def _positions(raw: Any) -> list[int]:
    """1-based reference numbers as the node face shows them, to 0-based indices."""
    out: list[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        if not text.isdigit() or int(text) < 1:
            raise ValueError(f"{text!r} is not a reference number. Use positions like: 2, 5")
        out.append(int(text) - 1)
    return sorted(set(out))


def _set_description(doc: cf.CharDoc, description: str) -> None:
    member = str(doc.manifest.text.get("path") or "text/description.md")
    data = description.encode("utf-8")
    doc.members[member] = data
    doc.manifest.text = {"path": member, "sha256": cf.sha256_bytes(data)}


class CompileReferencesRunner(NodeRunner):
    """One model's reference set: each reference resized to what that model accepts."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Compile References needs a character.")
        arch = str(node.params.get("arch") or encode.FLUX2_KLEIN_ARCH)

        def apply(doc: cf.CharDoc) -> None:
            encode.build_payload(doc.manifest, doc.members, _ref_images(doc), arch=arch)

        payload = Payload(arch=arch, kind=encode.PAYLOAD_REF, apply=apply)
        return NodeResult(outputs={"payload": payload})


class WriteCharacterRunner(NodeRunner):
    """Apply every payload to the identity and write one `.char` into models/characters."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Write .char needs a character.")
        doc = identity.doc
        for payload in inputs.get("payloads") or []:
            if isinstance(payload, Payload):
                payload.apply(doc)
        _apply_mode(doc, str(node.params.get("apply") or "auto"))
        # A typed name wins over the file it was loaded from: that is what Save as means.
        path = library.save(doc, _target_name(node.params.get("filename")) or identity.file or None)
        logger.info("Wrote %s with %d payload(s)", path.name, len(doc.manifest.payloads))
        for listener in _on_saved:
            listener(path.name)
        return NodeResult(outputs={"character": Identity(doc=cf.read(path), file=path.name)})


def _target_name(raw: Any) -> str | None:
    """The filename to write, inside models/characters. A path is reduced to its last part: a
    character written anywhere else is one the pickers cannot offer."""
    name = str(raw or "").strip().replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        return None
    return name if name.lower().endswith(".char") else f"{name}.char"


def _apply_mode(doc: cf.CharDoc, mode: str) -> None:
    """Which payload wins per model, for the models carrying both. `auto` clears the override."""
    for key in list(doc.manifest.payloads):
        if key.endswith(f"-{encode.PAYLOAD_LORA}"):
            continue
        if not encode.lora_payload(doc.manifest, key):
            continue
        if mode == "auto":
            doc.manifest.apply.pop(key, None)
        else:
            doc.manifest.apply[key] = mode


def _ref_images(doc: cf.CharDoc) -> list[Any]:
    """The character's own references decoded, so a payload compiles from truth not the library."""
    import io

    from PIL import Image

    images = []
    for ref in doc.manifest.refs:
        data = doc.members.get(str(ref.get("path") or ""))
        if data:
            with Image.open(io.BytesIO(data)) as handle:
                images.append(handle.convert("RGB").copy())
    return images


def _image_path(ref: Any) -> Any:
    """A wired image arrives as an AssetRef with a path; encoding reads files, not pixels."""
    from pathlib import Path

    path = getattr(ref, "path", None) or getattr(ref, "ref", None) or ref
    return Path(str(path))


class LoadCharacterRunner(NodeRunner):
    """A saved character by name, so a graph can apply one it did not build."""

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        chosen = str(node.params.get("file") or "").strip()
        if not chosen:
            raise ValueError("Pick a character.")
        path = library.resolve(chosen)
        if path is None:
            raise FileNotFoundError(
                f"Character {chosen!r} is not in models/characters/. Pick another."
            )
        return NodeResult(outputs={"character": Identity(doc=cf.read(path), file=path.name)})


class CharacterDatasetRunner(NodeRunner):
    """The character's references and description as a training set.

    Read from the `.char` rather than the asset library, because refs are truth and the library
    copies they came from may have been deleted or edited since.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Character to Dataset needs a character.")
        doc = identity.doc
        description = doc.members.get("text/description.md", b"").decode("utf-8")
        # The adapter binds to the description, so without one there is nothing for a prompt to
        # summon and the run trains a face nobody can address.
        if not description.strip():
            raise ValueError(
                "Give the character a description first: the adapter binds to it."
            )
        folder = _materialise(doc)
        images = [AssetRef(ref=str(p), path=str(p)) for p in folder]
        return NodeResult(outputs={"images": images, "captions": description})


class AttachAdapterRunner(NodeRunner):
    """File a trained adapter as this character's payload for one model.

    The base and rank come from the adapter's own safetensors metadata when it has any, because a
    LoRA loaded onto the wrong base degrades silently instead of raising.
    """

    def run(self, node: Node, inputs: dict[str, list[Any]], ctx: ExecutionContext) -> NodeResult:
        identity = _first(inputs.get("character"))
        if not isinstance(identity, Identity):
            raise ValueError("Attach Adapter needs a character.")
        lora = _first(inputs.get("lora"))
        stack = lora if isinstance(lora, tuple) else (lora,)
        ref = next((r for r in stack if getattr(r, "file", None)), None)
        if ref is None:
            raise ValueError("Attach Adapter needs a trained LoRA.")

        path = Path(str(ref.file))
        meta = _adapter_metadata(path)
        arch = str(node.params.get("arch") or meta.get("inline_arch") or "").strip()
        if not arch:
            raise ValueError(
                f"{path.name} does not say which model it trained on. Set Model on the node."
            )
        adapter = path.read_bytes()
        base = str(node.params.get("base") or meta.get("inline_base") or path.name)
        rank = _as_int(meta.get("inline_rank"), 16)
        steps = _as_int(meta.get("inline_steps"), 0)
        resolution = _as_int(meta.get("inline_resolution"), 512)

        def apply(doc: cf.CharDoc) -> None:
            encode.set_lora_payload(
                doc.manifest, doc.members, adapter, arch=arch,
                base=base, rank=rank, steps=steps, resolution=resolution,
            )

        payload = Payload(arch=arch, kind=encode.PAYLOAD_LORA, apply=apply)
        return NodeResult(outputs={"payload": payload})


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def _adapter_metadata(path: Path) -> dict[str, str]:
    """The adapter's safetensors header metadata, or {} for one trained before it was written."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return dict(handle.metadata() or {})
    except Exception:  # noqa: BLE001 - a missing label must not block attaching
        return {}


def _materialise(doc: cf.CharDoc) -> list[Path]:
    """The refs written out where a trainer can read them; the zip is not a path."""
    import tempfile

    folder = Path(tempfile.mkdtemp(prefix="char-dataset-"))
    written: list[Path] = []
    for index, ref in enumerate(doc.manifest.refs):
        data = doc.members.get(str(ref.get("path") or ""))
        if data:
            out = folder / f"{index:04d}.png"
            out.write_bytes(data)
            written.append(out)
    return written
