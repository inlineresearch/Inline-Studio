"""Character creation as a graph: identity once, a payload per model, one write."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from inline_core.characters import charfile as cf  # noqa: E402
from inline_core.characters import encode, library, weights  # noqa: E402
from inline_core.models.character.runner import (  # noqa: E402
    CompileReferencesRunner,
    EncodeCharacterRunner,
    Identity,
    WriteCharacterRunner,
)


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def encoders(tmp_path: Path) -> None:
    try:
        weights.ensure()
    except Exception as error:  # noqa: BLE001 - offline is a skip, not a failure
        pytest.skip(f"character encoders unavailable: {error}")


def _node(params: dict[str, object]):
    from inline_core.graph.schema import Node

    return Node(id="n", type="character/encode", params=params)


def _ctx() -> object:
    """A real context: the encode node reports its phases, so a None emitter is not the shape."""
    from inline_core.runtime.context import CancelToken, ExecutionContext
    from inline_core.runtime.progress import NullEmitter

    return ExecutionContext(
        run_id="r", policy=object(), emitter=NullEmitter(), cancel=CancelToken()  # type: ignore[arg-type]
    )


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (768, 1024), (180, 150, 140)).save(path)
    return path


def test_the_chain_writes_one_char_with_a_payload_per_model(
    tmp_path: Path, encoders: None
) -> None:
    """Encode once, compile per model, write once. Adding a model must not re-encode identity."""
    refs = [_image(tmp_path / "a.png"), _image(tmp_path / "b.png")]

    encoded = EncodeCharacterRunner().run(
        _node({"name": "Ada", "description": "green canvas jacket"}),
        {"images": refs},
        _ctx(),  # type: ignore[arg-type]
    )
    identity = encoded.outputs["character"]
    assert isinstance(identity, Identity)
    assert len(identity.doc.manifest.refs) == 2
    assert identity.doc.manifest.scoring.get("centroids"), "embeddings are the point of encoding"

    payloads = [
        CompileReferencesRunner()
        .run(_node({"arch": arch}), {"character": [identity]}, None)  # type: ignore[arg-type]
        .outputs["payload"]
        for arch in (encode.FLUX2_KLEIN_ARCH, "minimax-h3")
    ]

    written = WriteCharacterRunner().run(
        _node({}), {"character": [identity], "payloads": payloads}, _ctx(),  # type: ignore[arg-type]
    )
    saved = written.outputs["character"]
    assert isinstance(saved, Identity)

    path = library.resolve("Ada.char")
    assert path is not None
    doc = cf.read(path)
    assert set(doc.manifest.payloads) == {encode.FLUX2_KLEIN_ARCH, "minimax-h3"}
    assert encode.needs_rebuild(doc.manifest) is False
    # Identity was encoded once; both payloads share the same references.
    assert len(doc.manifest.refs) == 2


def test_encoding_refuses_before_the_encoders_are_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently pulling 385MB mid-graph is the wrong way to spend someone's bandwidth."""
    monkeypatch.setattr(weights, "present", lambda: False)
    with pytest.raises(ValueError, match="encoders"):
        EncodeCharacterRunner().run(
            _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx(),  # type: ignore[arg-type]
        )


def test_a_wired_description_beats_the_typed_one(tmp_path: Path, encoders: None) -> None:
    """So a Prompt node can drive the description that a trained adapter binds to."""
    result = EncodeCharacterRunner().run(
        _node({"name": "Ada", "description": "typed"}),
        {"images": [_image(tmp_path / "a.png")], "description": ["wired"]},
        _ctx(),  # type: ignore[arg-type]
    )
    doc = result.outputs["character"].doc
    assert doc.members["text/description.md"].decode() == "wired"


def test_load_reads_a_saved_character(tmp_path: Path, encoders: None) -> None:
    from inline_core.models.character.runner import LoadCharacterRunner

    doc = encode.char_encode([_image(tmp_path / "a.png")], name="Ada", description="green jacket")
    library.save(doc)

    loaded = LoadCharacterRunner().run(
        _node({"file": "Ada.char"}), {}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]
    assert isinstance(loaded, Identity)
    assert loaded.doc.manifest.name == "Ada"

    with pytest.raises(FileNotFoundError):
        LoadCharacterRunner().run(_node({"file": "Nobody.char"}), {}, None)  # type: ignore[arg-type]


def test_dataset_reads_refs_from_the_char_not_the_library(tmp_path: Path, encoders: None) -> None:
    """Refs are truth; the library copies they came from may since have been deleted or edited.

    It writes real dataset rows rather than loose images, because that is what Train LoRA runs
    from: the queue, the resumable checkpoint and the editable captions all hang off the rows.
    """
    from inline_core.models.character.runner import CharacterDatasetRunner
    from inline_core.models.training.runner import Dataset, TrainingBridge

    source = _image(tmp_path / "a.png")
    doc = encode.char_encode([source], name="Ada", description="green jacket")
    identity = Identity(doc=doc)
    source.unlink()  # the library copy is gone; the character must still yield a dataset

    staged: dict[str, object] = {}

    class _Training:
        def list_datasets(self) -> list[dict[str, object]]:
            return []

        def create_dataset(self, inp: dict[str, object]) -> dict[str, object]:
            staged["name"] = inp["name"]
            return {"id": "d1", "name": inp["name"]}

        def stage_from_path(self, path: str) -> list[dict[str, object]]:
            staged["folder"] = path
            return [{"assetId": "a1"}]

        def commit_staged(self, _did: str, rows: list[dict[str, object]]) -> list[dict[str, str]]:
            return [{"id": f"i{n}"} for n, _ in enumerate(rows)]

        def set_caption(self, item_id: str, caption: str) -> None:
            staged.setdefault("captions", []).append((item_id, caption))  # type: ignore[union-attr]

    out = CharacterDatasetRunner(TrainingBridge(_Training())).run(
        _node({}), {"character": [identity]}, _ctx(),  # type: ignore[arg-type]
    ).outputs

    assert out["dataset"] == Dataset(id="d1", name="Ada (character)")
    # Staged from the character's own refs, written out, not from the library path that is gone.
    assert Path(str(staged["folder"])).is_dir()
    assert staged["captions"] == [("i0", "green jacket")]


def test_attach_adapter_takes_its_base_from_the_adapter(tmp_path: Path, encoders: None) -> None:
    """A LoRA loaded onto the wrong base degrades silently, so the file says what it trained on."""
    import torch
    from safetensors.torch import save_file

    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.character.runner import AttachAdapterRunner, WriteCharacterRunner

    adapter = tmp_path / "ada.safetensors"
    save_file(
        {"w": torch.zeros(2, 2)},
        str(adapter),
        metadata={"inline_arch": "krea2", "inline_base": "krea2_turbo_bf16.safetensors",
                  "inline_rank": "16", "inline_steps": "600", "inline_resolution": "512"},
    )

    identity = Identity(doc=encode.char_encode([_image(tmp_path / "a.png")], name="Ada"))
    payload = AttachAdapterRunner().run(
        _node({}),
        {"character": [identity], "lora": [(LoraRef(file=str(adapter), strength=1.0),)]},
        _ctx(),  # type: ignore[arg-type]
    ).outputs["payload"]

    WriteCharacterRunner().run(
        _node({}), {"character": [identity], "payloads": [payload]}, _ctx(),  # type: ignore[arg-type]
    )
    saved = cf.read(library.resolve("Ada.char"))
    entry = saved.manifest.payloads["krea2-lora"]
    assert entry["base"] == "krea2_turbo_bf16.safetensors", "the base came from the file"
    assert entry["training"] == {"rank": 16, "steps": 600, "resolution": 512}


def test_attach_adapter_files_what_train_lora_hands_it(tmp_path: Path, encoders: None) -> None:
    """The wire from Train LoRA is the whole point of the node, and it raised twice: the trainer
    emitted a bare path where every `lora` input reads a LoraRef stack, and then emitted one
    relative to the models root, which opened under the server's CWD and was not there."""
    import torch
    from safetensors.torch import save_file

    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.character.runner import AttachAdapterRunner

    adapter = tmp_path / "trained.safetensors"
    save_file(
        {"w": torch.zeros(2, 2)},
        str(adapter),
        metadata={"inline_arch": "flux2-klein", "inline_base": "flux2_base.safetensors",
                  "inline_rank": "32", "inline_steps": "1200", "inline_resolution": "1024"},
    )
    identity = Identity(doc=encode.char_encode([_image(tmp_path / "a.png")], name="Ada"))

    # The stack the trainer now emits, and the bare path an older cached run left behind. Both
    # absolute: a relative one reads as missing metadata rather than as a missing file.
    for wired in ((LoraRef(file=str(adapter), strength=1.0),), str(adapter)):
        payload = AttachAdapterRunner().run(
            _node({}),
            {"character": [identity], "lora": [wired]},
            _ctx(),  # type: ignore[arg-type]
        ).outputs["payload"]
        assert payload.arch == "flux2-klein", "the settings came off the wired adapter"


def test_the_adapter_strength_rides_in_the_file(tmp_path: Path, encoders: None) -> None:
    """An overfit adapter is only usable turned down, and the character wire carries no controls:
    a generation node fused every character at 1.0 with nowhere to say otherwise."""
    import torch
    from safetensors.torch import save_file

    from inline_core.characters import apply as characters
    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.character.runner import AttachAdapterRunner, WriteCharacterRunner

    adapter = tmp_path / "soft.safetensors"
    save_file(
        {"w": torch.zeros(2, 2)},
        str(adapter),
        metadata={"inline_arch": "krea2", "inline_base": "krea2_raw_bf16.safetensors"},
    )
    identity = Identity(doc=encode.char_encode([_image(tmp_path / "a.png")], name="Soft"))

    payload = AttachAdapterRunner().run(
        _node({"strength": 0.5}),
        {"character": [identity], "lora": [(LoraRef(file=str(adapter), strength=1.0),)]},
        _ctx(),  # type: ignore[arg-type]
    ).outputs["payload"]
    WriteCharacterRunner().run(
        _node({}), {"character": [identity], "payloads": [payload]}, _ctx(),  # type: ignore[arg-type]
    )

    saved = cf.read(library.resolve("Soft.char"))
    assert saved.manifest.payloads["krea2-lora"]["strength"] == 0.5
    # And it reaches the runner, which is the half that was hardcoded.
    assert characters.char_apply("Soft.char", "krea2").lora_strength == 0.5


def test_a_character_filed_before_strength_existed_still_fuses_at_one(
    tmp_path: Path, encoders: None
) -> None:
    doc = encode.char_encode([_image(tmp_path / "a.png")], name="Old")
    assert encode.lora_strength(doc.manifest, "krea2") == 1.0


def test_an_adapter_without_provenance_needs_its_model(tmp_path: Path, encoders: None) -> None:
    """Older adapters carry no metadata; guessing the architecture is how you get a wrong face."""
    import torch
    from safetensors.torch import save_file

    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.character.runner import AttachAdapterRunner

    adapter = tmp_path / "old.safetensors"
    save_file({"w": torch.zeros(2, 2)}, str(adapter))
    identity = Identity(doc=encode.char_encode([_image(tmp_path / "a.png")], name="Ada"))

    with pytest.raises(ValueError, match="which model"):
        AttachAdapterRunner().run(
            _node({}),
            {"character": [identity], "lora": [(LoraRef(file=str(adapter), strength=1.0),)]},
            _ctx(),  # type: ignore[arg-type]
        )


def test_a_relative_adapter_path_is_named_as_missing(tmp_path: Path, encoders: None) -> None:
    """An adapter that cannot be opened is not an adapter with no provenance. Saying the second sent
    a user looking for a Model setting when the path was resolved against the wrong root."""
    from inline_core.graph.loader_runners import LoraRef
    from inline_core.models.character.runner import AttachAdapterRunner

    identity = Identity(doc=encode.char_encode([_image(tmp_path / "a.png")], name="Ada"))
    with pytest.raises(ValueError, match="cannot be read"):
        AttachAdapterRunner().run(
            _node({}),
            {
                "character": [identity],
                "lora": [(LoraRef(file="loras/gone.safetensors", strength=1.0),)],
            },
            _ctx(),  # type: ignore[arg-type]
        )


def test_the_starter_chain_validates_against_the_descriptors() -> None:
    """The canvas drops this exact chain pre-wired, naming ports by id. A renamed port would fail
    at submit for the user and pass every unit test on both sides, so pin the ids here."""
    from inline_core.graph.registry import build_default_registry
    from inline_core.graph.schema import parse_graph
    from inline_core.graph.validate import validate
    from inline_core.models.character.runner import register_character_nodes

    registry = build_default_registry()
    register_character_nodes(registry)
    graph = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "img", "type": "input/image",
                 "params": {"asset": {"ref": "path", "path": "/tmp/a.png"}}},
                {"id": "enc", "type": "character/encode", "params": {"name": "Ada"},
                 "inputs": {"images": [{"from": "img", "output": "image"}]}},
                {"id": "refs", "type": "character/references", "params": {},
                 "inputs": {"character": [{"from": "enc", "output": "character"}]}},
                {"id": "write", "type": "character/write", "params": {},
                 "inputs": {
                     "character": [{"from": "enc", "output": "character"}],
                     "payloads": [{"from": "refs", "output": "payload"}],
                 }},
            ],
        }
    )
    validate(graph, "write", registry)


def _edit(params: dict[str, object]) -> object:
    from inline_core.graph.schema import Node

    return Node(id="e", type="character/edit", params=params)


def test_editing_adds_and_drops_references_without_an_encoder(
    tmp_path: Path, encoders: None
) -> None:
    """The point of editing in place: no encoder runs, so it is instant however big the set."""
    from inline_core.models.character.runner import EditCharacterRunner

    refs = [_image(tmp_path / f"{n}.png") for n in "abc"]
    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": refs}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    extra = _image(tmp_path / "d.png")
    edited = EditCharacterRunner().run(
        _edit({"drop": "2"}), {"character": [identity], "images": [extra]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    assert isinstance(edited, Identity)
    assert len(edited.doc.manifest.refs) == 3, "three minus one dropped, plus one added"
    assert len(identity.doc.manifest.refs) == 3, "the upstream node's cached output is untouched"


def test_editing_renames_and_redescribes(tmp_path: Path, encoders: None) -> None:
    from inline_core.models.character.runner import EditCharacterRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada", "description": "green jacket"}),
        {"images": [_image(tmp_path / "a.png")]},
        _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    edited = EditCharacterRunner().run(
        _edit({"name": "Ada Lovelace", "description": "red jacket"}),
        {"character": [identity]},
        _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    assert edited.doc.manifest.name == "Ada Lovelace"
    member = str(edited.doc.manifest.text["path"])
    assert edited.doc.members[member].decode() == "red jacket"


def test_a_blank_edit_changes_nothing(tmp_path: Path, encoders: None) -> None:
    """Empty fields mean keep, not clear: the node is reached for one field at a time."""
    from inline_core.models.character.runner import EditCharacterRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada", "description": "green jacket"}),
        {"images": [_image(tmp_path / "a.png")]},
        _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    edited = EditCharacterRunner().run(
        _edit({}), {"character": [identity]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    assert edited.doc.manifest.name == "Ada"
    assert len(edited.doc.manifest.refs) == 1


def test_a_reference_number_that_is_not_a_number_is_refused(
    tmp_path: Path, encoders: None
) -> None:
    """Silently ignoring it would drop nothing and report success, reading as a broken button."""
    from inline_core.models.character.runner import EditCharacterRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]

    with pytest.raises(ValueError, match="reference number"):
        EditCharacterRunner().run(
            _edit({"drop": "second"}), {"character": [identity]}, _ctx(),  # type: ignore[arg-type]
        )


def test_writing_an_edited_character_replaces_it_rather_than_forking_it(
    tmp_path: Path, encoders: None
) -> None:
    """A loaded character keeps its filename: it is what every node already picking it stores."""
    from inline_core.models.character.runner import EditCharacterRunner, LoadCharacterRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]
    WriteCharacterRunner().run(
        _node({}), {"character": [identity], "payloads": []}, _ctx(),  # type: ignore[arg-type]
    )
    before = sorted(p.name for p in library.root().glob("*.char"))

    loaded = LoadCharacterRunner().run(
        _node({"file": "Ada.char"}), {}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]
    edited = EditCharacterRunner().run(
        _edit({"name": "Ada L"}), {"character": [loaded]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]
    WriteCharacterRunner().run(
        _node({}), {"character": [edited], "payloads": []}, _ctx(),  # type: ignore[arg-type]
    )

    assert sorted(p.name for p in library.root().glob("*.char")) == before
    assert cf.read(library.resolve("Ada.char")).manifest.name == "Ada L"


def test_write_can_force_references_over_an_adapter(tmp_path: Path, encoders: None) -> None:
    """A character holding both is the one case the file cannot decide alone: the default is the
    adapter, and this is how a user asks for the references instead."""
    from inline_core.models.character.runner import Payload

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]
    refs = CompileReferencesRunner().run(
        _node({"arch": encode.FLUX2_KLEIN_ARCH}), {"character": [identity]}, _ctx(),  # type: ignore[arg-type]
    ).outputs["payload"]

    def fake_lora(doc: cf.CharDoc) -> None:
        key = encode.payload_key(encode.FLUX2_KLEIN_ARCH, encode.PAYLOAD_LORA)
        doc.manifest.payloads[key] = {"policy": {}, "source_sha256": "", "files": []}

    adapter = Payload(arch=encode.FLUX2_KLEIN_ARCH, kind=encode.PAYLOAD_LORA, apply=fake_lora)

    saved = WriteCharacterRunner().run(
        _node({"apply": "reference"}),
        {"character": [identity], "payloads": [refs, adapter]},
        _ctx(),  # type: ignore[arg-type]
    ).outputs["character"]
    doc = cf.read(library.resolve("Ada.char"))
    assert doc.manifest.apply[encode.FLUX2_KLEIN_ARCH] == "reference"

    # Re-written from the saved character, so it lands on the same file rather than forking.
    WriteCharacterRunner().run(
        _node({"apply": "auto"}),
        {"character": [saved], "payloads": [refs, adapter]},
        _ctx(),  # type: ignore[arg-type]
    )
    doc = cf.read(library.resolve("Ada.char"))
    assert encode.FLUX2_KLEIN_ARCH not in doc.manifest.apply, "auto clears the override"


def test_encoding_refuses_with_no_references(tmp_path: Path, encoders: None) -> None:
    with pytest.raises(ValueError, match="at least one face reference"):
        EncodeCharacterRunner().run(_node({"name": "Ada"}), {"images": []}, _ctx())  # type: ignore[arg-type]


def test_body_and_clothing_alone_are_not_a_character(tmp_path: Path, encoders: None) -> None:
    """They condition on top of an identity. Encoding from them would produce a character whose
    likeness nothing carries, and the face is what every model's identity signal comes from."""
    with pytest.raises(ValueError, match="at least one face reference"):
        EncodeCharacterRunner().run(  # type: ignore[arg-type]
            _node({"name": "Ada"}), {"images": [], "body": ["b"], "cloth": ["c"]}, _ctx()
        )


def test_the_last_reference_cannot_be_dropped(tmp_path: Path, encoders: None) -> None:
    """A character with no references is not a character, so the edit is refused, not silently
    ignored."""
    from inline_core.models.character.runner import EditCharacterRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx()  # type: ignore[arg-type]
    ).outputs["character"]

    with pytest.raises(ValueError, match="at least one reference"):
        EditCharacterRunner().run(
            _edit({"drop": "1"}), {"character": [identity]}, _ctx()  # type: ignore[arg-type]
        )


def test_encoding_reports_its_phases(tmp_path: Path, encoders: None) -> None:
    """Two embedding passes over every reference: silence reads as a hung node."""
    from inline_core.runtime.context import CancelToken, ExecutionContext
    from inline_core.runtime.progress import CollectingEmitter

    emitter = CollectingEmitter()
    ctx = ExecutionContext(
        run_id="r", policy=object(), emitter=emitter, cancel=CancelToken()  # type: ignore[arg-type]
    )
    EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, ctx
    )
    assert emitter.events, "the encode reported nothing"
    assert any(e.status for e in emitter.events)


def test_a_dataset_needs_the_character_to_have_a_description(
    tmp_path: Path, encoders: None
) -> None:
    """The adapter binds to the description; training without one produces a face no prompt can
    summon, which only shows up an hour later in the render."""
    from inline_core.models.character.runner import CharacterDatasetRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx()  # type: ignore[arg-type]
    ).outputs["character"]

    with pytest.raises(ValueError, match="description"):
        CharacterDatasetRunner().run(_node({}), {"character": [identity]}, _ctx())  # type: ignore[arg-type]


def test_save_as_names_the_file_and_beats_the_one_it_was_loaded_from(
    tmp_path: Path, encoders: None
) -> None:
    """Save as is how a character is forked on purpose, so it wins over the loaded name."""
    from inline_core.models.character.runner import LoadCharacterRunner

    identity = EncodeCharacterRunner().run(
        _node({"name": "Ada"}), {"images": [_image(tmp_path / "a.png")]}, _ctx()  # type: ignore[arg-type]
    ).outputs["character"]
    WriteCharacterRunner().run(
        _node({}), {"character": [identity], "payloads": []}, _ctx()  # type: ignore[arg-type]
    )
    loaded = LoadCharacterRunner().run(
        _node({"file": "Ada.char"}), {}, _ctx()  # type: ignore[arg-type]
    ).outputs["character"]

    WriteCharacterRunner().run(
        _node({"filename": "Ada v2"}), {"character": [loaded], "payloads": []}, _ctx()  # type: ignore[arg-type]
    )
    names = sorted(p.name for p in library.root().glob("*.char"))
    assert names == ["Ada v2.char", "Ada.char"], "the original survives, the copy is named"


def test_save_as_keeps_the_character_inside_the_library(tmp_path: Path, encoders: None) -> None:
    """Written elsewhere it is a character no picker can offer, so a path becomes a name."""
    from inline_core.models.character.runner import _target_name

    assert _target_name("  Ada  ") == "Ada.char"
    assert _target_name("Ada.char") == "Ada.char"
    assert _target_name("/etc/passwd") == "passwd.char"
    assert _target_name("../../escape.char") == "escape.char"
    assert _target_name("") is None



# --- verify-refs ---------------------------------------------------------------------------------


def _fake_faces(monkeypatch: pytest.MonkeyPatch, vectors: list[list[float]]) -> None:
    """Deterministic identity, so the node's own logic is what is under test and not a detector.

    Keyed on the reference's own pixels, not call order: freezing re-decodes the members it froze
    over, so an order-keyed stub would hand those a different vector than the pass that flagged.
    """
    from inline_core.characters import scoring, weights

    monkeypatch.setattr(weights, "present", lambda: True)
    by_colour = {_REF_COLOURS[i]: v for i, v in enumerate(vectors)}

    def face(image: object) -> list[float] | None:
        return by_colour.get(image.getpixel((0, 0))) or None  # type: ignore[attr-defined]

    monkeypatch.setattr(scoring, "embed_face", face)
    monkeypatch.setattr(scoring, "embed_subject", lambda _image: [1.0, 0.0, 0.0])


#: One flat colour per reference slot, so a stub can identify a reference by its own pixels.
_REF_COLOURS = [(index * 30 + 10, 90, 140) for index in range(12)]


def _character(tmp_path: Path, count: int, name: str = "Ada") -> Identity:
    from inline_core.characters import charfile as cf
    from inline_core.characters import encode

    manifest = cf.Manifest(char_id="c", name=name, created_at=0, modified_at=0)
    members: dict[str, bytes] = {}
    for index in range(count):
        image = Image.new("RGB", (320, 320), _REF_COLOURS[index])
        member = cf.member_name("refs", index, ".png")
        data = encode._png_bytes(image)
        members[member] = data
        manifest.refs.append(
            {"path": member, "sha256": cf.sha256_bytes(data), "width": 320, "height": 320,
             "origin": cf.ORIGIN_ORIGINAL}
        )
    return Identity(doc=cf.CharDoc(manifest=manifest, members=members))


def _verify(identity: Identity, **params: object):
    from inline_core.graph.schema import Node
    from inline_core.models.character.runner import VerifyReferencesRunner

    merged: dict[str, object] = {"on_outlier": "flag", "floor": 25.0}
    merged.update(params)
    return VerifyReferencesRunner().run(
        Node(id="v", type="character/verify-refs", params=merged), {"character": [identity]}, _ctx()
    ).outputs["character"]


_SAME = [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.98, 0.0, 0.1], [0.99, 0.05, 0.05]]
_OTHER = [0.0, 1.0, 0.0]


def test_verify_flags_a_reference_of_someone_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-match scoring means one wrong reference is a backdoor into every take."""
    _fake_faces(monkeypatch, [*_SAME[:3], _OTHER])
    out = _verify(_character(tmp_path, 4))

    verdict = out.doc.manifest.scoring["verification"]
    assert verdict["mode"] == "bootstrap"
    assert verdict["flagged"] == [3]
    assert len(out.doc.manifest.refs) == 4, "flag is the default and must remove nothing"


def test_verify_freezes_the_originals_once_it_has_checked_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inline_core.characters import encode

    _fake_faces(monkeypatch, [*_SAME[:3], _OTHER])
    first = _verify(_character(tmp_path, 4))
    assert encode.originals_frozen(first.doc.manifest)
    frozen = dict(first.doc.manifest.scoring["originals"])

    _fake_faces(monkeypatch, [*_SAME[:3], _OTHER])
    again = _verify(Identity(doc=first.doc))
    assert again.doc.manifest.scoring["verification"]["mode"] == "existing"
    assert again.doc.manifest.scoring["originals"] == frozen, "the identity target moved"


def test_verify_quarantines_rather_than_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refs are truth and the file one came from may be long gone, so removal stays reversible."""
    _fake_faces(monkeypatch, [*_SAME[:3], _OTHER])
    out = _verify(_character(tmp_path, 4), on_outlier="quarantine")

    assert len(out.doc.manifest.refs) == 3
    kept = [m for m in out.doc.members if m.startswith("quarantined/")]
    assert len(kept) == 1, "the removed reference's bytes were not kept"


def test_verify_never_removes_a_reference_with_no_face(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Those are the wide and full-body shots the hints ask for, and the only ones that let the
    subject term speak to a wide take at all."""
    _fake_faces(monkeypatch, [*_SAME[:3], []])
    out = _verify(_character(tmp_path, 4), on_outlier="quarantine")

    verdict = out.doc.manifest.scoring["verification"]
    assert verdict["unchecked"] == [3]
    assert verdict["flagged"] == []
    assert len(out.doc.manifest.refs) == 4


def test_verify_declines_to_flag_a_set_too_small_to_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With two, "agreement with the others" is one pairwise number and cannot say which is odd."""
    _fake_faces(monkeypatch, [_SAME[0], _OTHER])
    out = _verify(_character(tmp_path, 2), on_outlier="quarantine")

    verdict = out.doc.manifest.scoring["verification"]
    assert verdict["flagged"] == []
    assert "odd one out" in verdict["note"]
    assert len(out.doc.manifest.refs) == 2


def test_verify_removes_a_byte_identical_duplicate_even_in_flag_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duplicate is not a judgement call: it doubles that image's weight in a training mix and
    spends a reference slot the model addresses by position."""
    from inline_core.characters import charfile as cf

    # Three vectors for three colours: the twin shares reference 0's pixels, so it shares its face.
    _fake_faces(monkeypatch, _SAME[:3])
    identity = _character(tmp_path, 3)
    doc = identity.doc
    twin = dict(doc.manifest.refs[0])
    twin["path"] = cf.member_name("refs", 9, ".png")
    doc.members[twin["path"]] = doc.members[doc.manifest.refs[0]["path"]]
    doc.manifest.refs.append(twin)

    out = _verify(identity)

    verdict = out.doc.manifest.scoring["verification"]
    assert verdict["removed"]["duplicates"] == ["refs/009.png"]
    # Emptied, not left at [3]: the positions describe the set that survived, not the one checked.
    assert verdict["duplicates"] == []
    assert len(out.doc.manifest.refs) == 3


def test_a_stored_verdict_names_positions_in_the_set_that_survived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every list in the verdict is a position into `manifest.refs`, and removing one shifts each
    position after it - so a report stored as found would ring a reference that is now another."""
    _fake_faces(monkeypatch, [_SAME[0], _SAME[1], _SAME[2], _OTHER])
    identity = _character(tmp_path, 4)
    doc = identity.doc
    # A duplicate of reference 0 in the middle, so removing it shifts the flagged impostor down.
    twin = dict(doc.manifest.refs[0])
    twin["path"] = cf.member_name("refs", 9, ".png")
    doc.members[twin["path"]] = doc.members[doc.manifest.refs[0]["path"]]
    doc.manifest.refs.insert(1, twin)

    verdict = _verify(identity).doc.manifest.scoring["verification"]

    assert len(verdict["agreement"]) == 4, "the report still describes five references"
    assert verdict["flagged"] == [3], "the impostor kept the position it held before the dedup"


def test_write_refuses_a_payload_built_from_a_different_reference_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compile References uses its `character` input only to read settings - it compiles from the
    doc Write hands it - so wiring Write ahead of the verify node would save the unchecked set."""
    from inline_core.graph.schema import Node
    from inline_core.models.character.runner import CompileReferencesRunner, WriteCharacterRunner

    _fake_faces(monkeypatch, [*_SAME[:3], _OTHER])
    unchecked = _character(tmp_path, 4)
    verified = _verify(Identity(doc=copy.deepcopy(unchecked.doc)), on_outlier="quarantine")

    payload = CompileReferencesRunner().run(
        Node(id="p", type="character/references", params={}), {"character": [verified]}, _ctx()
    ).outputs["payload"]

    with pytest.raises(ValueError, match="different version"):
        WriteCharacterRunner().run(
            Node(id="w", type="character/write", params={"filename": "Ada"}),
            {"character": [unchecked], "payloads": [payload]},
            _ctx(),
        )


def test_a_verified_drop_reaches_the_training_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LoRA path matters more than the reference path: a bad reference bakes into the weights.

    `commit_staged` reconciles the dataset against exactly what was staged, so a reference the
    verify node took out is a row the next run removes rather than one left behind.
    """
    from inline_core.models.character.runner import CharacterDatasetRunner
    from inline_core.models.training.runner import TrainingBridge

    _fake_faces(monkeypatch, [*_SAME[:3], _OTHER])
    identity = _character(tmp_path, 4)
    identity.doc.members["text/description.md"] = b"green jacket"
    identity.doc.manifest.text = {"path": "text/description.md", "sha256": ""}
    verified = _verify(identity, on_outlier="quarantine")

    staged: dict[str, object] = {}

    class _Training:
        def list_datasets(self) -> list[dict[str, object]]:
            return []

        def create_dataset(self, inp: dict[str, object]) -> dict[str, object]:
            return {"id": "d1", "name": inp["name"]}

        def stage_from_path(self, path: str) -> list[dict[str, object]]:
            staged["files"] = sorted(p.name for p in Path(path).iterdir())
            return [{"assetId": f"a{n}"} for n, _ in enumerate(staged["files"])]  # type: ignore[arg-type]

        def commit_staged(self, _did: str, rows: list[dict[str, object]]) -> list[dict[str, str]]:
            return [{"id": f"i{n}"} for n, _ in enumerate(rows)]

        def set_caption(self, item_id: str, caption: str) -> None:
            return None

    CharacterDatasetRunner(TrainingBridge(_Training())).run(
        _node({}), {"character": [verified]}, _ctx(),  # type: ignore[arg-type]
    )

    assert staged["files"] == ["0000.png", "0001.png", "0002.png"], "the dropped ref still trained"


def test_a_character_that_never_harvested_is_unchanged_by_the_feature(tmp_path: Path) -> None:
    """The loop is opt-in and additive: a character built without it must compile the same bytes
    and stage the same training rows as one built before it existed."""
    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    members: dict[str, bytes] = {}
    for index in range(3):
        image = Image.new("RGB", (320, 320), (index * 30 + 10, 90, 140))
        member = cf.member_name("refs", index, ".png")
        data = encode._png_bytes(image)
        members[member] = data
        # No origin field at all, the way every character written before this was.
        manifest.refs.append({"path": member, "sha256": cf.sha256_bytes(data)})
    doc = cf.CharDoc(manifest=manifest, members=members)

    encode.build_payload(manifest, members, encode.ref_images(doc))
    entry = manifest.payloads[encode.FLUX2_KLEIN_ARCH]

    assert entry["harvested_count"] == 0
    assert [f["path"] for f in entry["files"]] == [
        f"payloads/flux2-klein/ref_{i:03d}.png" for i in range(3)
    ]
    assert encode.harvested(manifest) == []
    assert len(encode.originals(manifest)) == 3


def test_the_harvest_canvas_graph_validates_against_the_registered_descriptors(
    tmp_path: Path,
) -> None:
    """The harvest chain is only real if the port ids the canvas emits are the ones the nodes
    declare. A mismatch is a run that dies at submit, and neither side's unit tests would see it."""
    import sqlite3

    from inline_core.graph.registry import build_default_registry
    from inline_core.graph.schema import parse_graph
    from inline_core.graph.validate import validate
    from inline_core.models.character.runner import register_character_nodes
    from inline_core.studio import moodboard as mb
    from inline_core.studio.graph_build import build_workflow_graph
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")

    conn.execute(
        "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
        "VALUES ('take', 'p', 'take', 'assets/take.png', 'image', 0)"
    )
    _image(tmp_path / "assets" / "take.png")
    asset = mb.add_asset(conn, "take", 0, 0)
    load = mb.add_core_node(conn, "character/load", 0, 0)
    ingest = mb.add_core_node(conn, "character/ingest-approved", 0, 0)
    write = mb.add_core_node(conn, "character/write", 0, 0)
    mb.create_connector(conn, load["id"], ingest["id"], "character", "character")
    mb.create_connector(conn, asset["id"], ingest["id"], "image", "image")
    mb.create_connector(conn, ingest["id"], write["id"], "character", "character")

    # The default registry, because the take reaches the node as an `input/image` source node.
    registry = build_default_registry()
    register_character_nodes(registry)
    graph_dict, target = build_workflow_graph(conn, tmp_path, write["id"], lambda _t, _p: False)
    validate(parse_graph(graph_dict), target, registry)
