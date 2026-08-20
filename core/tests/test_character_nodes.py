"""Character creation as a graph: identity once, a payload per model, one write."""

from __future__ import annotations

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
    with pytest.raises(ValueError, match="at least one reference"):
        EncodeCharacterRunner().run(_node({"name": "Ada"}), {"images": []}, _ctx())  # type: ignore[arg-type]


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

