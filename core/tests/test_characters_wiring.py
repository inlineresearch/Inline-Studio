"""Characters wired into Core: the catalog category, the node param, and cache invalidation."""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.characters import charfile as cf
from inline_core.graph.cache import asset_content_hashes
from inline_core.graph.schema import Graph, Node
from inline_core.models.catalog import CATEGORIES, ModelCatalog


def _char_bytes(description: str = "green jacket") -> bytes:
    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    manifest.refs.append({"path": "refs/000.png", "sha256": cf.sha256_bytes(b"x")})
    manifest.text = {"path": "text/description.md", "sha256": cf.sha256_bytes(description.encode())}
    doc = cf.CharDoc(
        manifest=manifest,
        members={"refs/000.png": b"x", "text/description.md": description.encode()},
    )
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(cf.MANIFEST_NAME, cf.dumps_manifest(doc.manifest))
        for name, data in doc.members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# --- catalog -------------------------------------------------------------------------------------


def test_characters_is_a_catalog_category() -> None:
    assert "characters" in CATEGORIES


def test_a_char_is_listed_under_characters_and_nowhere_else(tmp_path: Path) -> None:
    """A `.char` in a diffusion dropdown would be a checkpoint the loader cannot open."""
    root = tmp_path / "models"
    (root / "characters").mkdir(parents=True)
    (root / "diffusion_models").mkdir(parents=True)
    (root / "characters" / "Ada.char").write_bytes(_char_bytes())
    (root / "diffusion_models" / "Stray.char").write_bytes(_char_bytes())
    (root / "diffusion_models" / "real.safetensors").write_bytes(b"weights")

    catalog = ModelCatalog(root)
    catalog.rescan()
    assert catalog.list("characters") == ["Ada.char"]
    assert catalog.list("diffusion_models") == ["real.safetensors"]


def test_a_weight_dropped_into_characters_is_not_listed(tmp_path: Path) -> None:
    root = tmp_path / "models"
    (root / "characters").mkdir(parents=True)
    (root / "characters" / "loose.safetensors").write_bytes(b"weights")
    catalog = ModelCatalog(root)
    catalog.rescan()
    assert catalog.list("characters") == []


def test_a_folder_in_characters_is_not_listed_as_one(tmp_path: Path) -> None:
    root = tmp_path / "models"
    nested = root / "characters" / "somedir"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"weights")
    catalog = ModelCatalog(root)
    catalog.rescan()
    assert catalog.list("characters") == []


def test_adding_a_character_moves_the_catalog_fingerprint(tmp_path: Path) -> None:
    """The fingerprint feeds the registry version, which is what refreshes the node dropdown."""
    root = tmp_path / "models"
    (root / "characters").mkdir(parents=True)
    catalog = ModelCatalog(root)
    catalog.rescan()
    before = catalog.fingerprint()
    (root / "characters" / "Ada.char").write_bytes(_char_bytes())
    catalog.rescan()
    assert catalog.fingerprint() != before


# --- the node param ------------------------------------------------------------------------------


def test_the_flux2_node_offers_a_character_dropdown() -> None:
    runner = pytest.importorskip("inline_core.models.flux2.runner")
    field = next(p for p in runner.FLUX2.params if p.key == "character")
    assert field.options_from == "characters"
    assert field.default == ""
    # Not advanced: picking a character is the point of the feature, not a power-user escape hatch.
    assert field.advanced is False


# --- cache invalidation --------------------------------------------------------------------------


def _graph_with_character(chosen: str) -> Graph:
    return Graph(
        schema_version=1,
        nodes=[Node(id="gen", type="black-forest-labs/flux-2", params={"character": chosen})],
    )


def test_editing_a_character_in_place_invalidates_the_node_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filename it is picked by does not change, so without a content hash the cache would
    serve a take of the previous face."""
    root = tmp_path / "models"
    (root / "characters").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.chdir(tmp_path)

    path = root / "characters" / "Ada.char"
    path.write_bytes(_char_bytes("green jacket"))
    graph = _graph_with_character("Ada.char")
    before = asset_content_hashes(graph)["gen"]

    path.write_bytes(_char_bytes("red jacket"))
    assert asset_content_hashes(graph)["gen"] != before


def test_no_character_contributes_no_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    assert asset_content_hashes(_graph_with_character("")) == {}
    assert asset_content_hashes(_graph_with_character("Missing.char")) == {}


# --- the runner's own character step --------------------------------------------------------------


def _install_character(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refs: int) -> None:
    """A real, valid character in a real models root, so the runner resolves it for real."""
    pytest.importorskip("PIL")
    from PIL import Image

    from inline_core.characters import encode, library

    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)

    manifest = cf.Manifest(char_id="c", name="Ada", created_at=0, modified_at=0)
    members: dict[str, bytes] = {}
    images = []
    for index in range(refs):
        image = Image.new("RGB", (640, 480), (index * 30, 90, 140))
        images.append(image)
        member = cf.member_name("refs", index, ".png")
        data = encode._png_bytes(image)
        members[member] = data
        manifest.refs.append({"path": member, "sha256": cf.sha256_bytes(data)})
    members["text/description.md"] = b"short brown hair"
    manifest.text = {"path": "text/description.md", "sha256": cf.sha256_bytes(b"short brown hair")}
    encode.build_payload(manifest, members, images)
    library.save(cf.CharDoc(manifest=manifest, members=members))


def test_the_runner_appends_character_refs_after_the_wired_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appending is what keeps the numbered strip on the node face equal to the numbers the prompt
    resolves: the strip is built client-side from connectors and cannot see injected refs."""
    flux2 = pytest.importorskip("inline_core.models.flux2.runner")
    _install_character(tmp_path, monkeypatch, refs=2)

    # Two references the user wired themselves already hold positions 1 and 2.
    applied = flux2._apply_character({"character": "Ada.char"}, wired=2)
    assert applied is not None
    assert len(applied.refs) == 2
    assert applied.prefix.startswith("Images 3 and 4 show Ada")
    assert "short brown hair" in applied.prefix


def test_the_runner_applies_nothing_when_no_character_is_picked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flux2 = pytest.importorskip("inline_core.models.flux2.runner")
    _install_character(tmp_path, monkeypatch, refs=1)
    assert flux2._apply_character({}, wired=0) is None
    assert flux2._apply_character({"character": ""}, wired=0) is None
    assert flux2._apply_character({"character": "   "}, wired=0) is None


def test_the_runner_refuses_a_character_that_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendering without the character would produce a plausible image of the wrong person."""
    flux2 = pytest.importorskip("inline_core.models.flux2.runner")
    _install_character(tmp_path, monkeypatch, refs=1)
    with pytest.raises(FileNotFoundError):
        flux2._apply_character({"character": "Nobody.char"}, wired=0)
