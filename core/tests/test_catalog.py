from __future__ import annotations

from pathlib import Path

from inline_core.graph.descriptor import NodeDescriptor, ParamField, Widget
from inline_core.models.catalog import ModelCatalog
from inline_core.server.serialize import descriptor_json


def test_catalog_scans_weight_files_by_category(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    (tmp_path / "vae" / "flux-vae.safetensors").write_bytes(b"x")
    (tmp_path / "diffusion_models" / "z-image.safetensors").write_bytes(b"y")
    (tmp_path / "loras" / "readme.txt").write_text("not a weight file")

    scan = catalog.rescan()

    assert scan["vae"] == ["flux-vae.safetensors"]
    assert scan["diffusion_models"] == ["z-image.safetensors"]
    assert catalog.list("loras") == []


def test_catalog_fingerprint_changes_when_a_model_is_added(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    before = catalog.fingerprint()

    (tmp_path / "checkpoints" / "sdxl.safetensors").write_bytes(b"z")
    catalog.rescan()

    assert catalog.fingerprint() != before


def test_descriptor_options_come_from_catalog(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    (tmp_path / "vae" / "flux-vae.safetensors").write_bytes(b"x")
    catalog.rescan()
    descriptor = NodeDescriptor(
        type="m",
        title="M",
        category="Image",
        params=(ParamField("vae", "VAE", Widget.SELECT, "", options_from="vae"),),
    )

    param = descriptor_json(descriptor, catalog)["params"][0]

    assert param["optionsFrom"] == "vae"
    assert {"value": "flux-vae.safetensors", "label": "flux-vae.safetensors"} in param["options"]


def test_catalog_lists_model_folders(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    folder = tmp_path / "text_encoders" / "qwen3-4b"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"weights")

    scan = catalog.rescan()

    # the folder is listed once; its inner weight file is not listed separately
    assert scan["text_encoders"] == ["qwen3-4b"]


def test_catalog_merges_several_roots(tmp_path: Path) -> None:
    """A custom --models-dir must not hide ./models: both roots are scanned."""
    primary, extra = tmp_path / "primary", tmp_path / "extra"
    (primary / "loras").mkdir(parents=True)
    (extra / "loras").mkdir(parents=True)
    (primary / "loras" / "a.safetensors").write_bytes(b"a")
    (extra / "loras" / "b.safetensors").write_bytes(b"b")

    scan = ModelCatalog([primary, extra]).rescan()

    assert scan["loras"] == ["a.safetensors", "b.safetensors"]


def test_a_name_in_two_roots_is_listed_once(tmp_path: Path) -> None:
    primary, extra = tmp_path / "primary", tmp_path / "extra"
    (primary / "vae").mkdir(parents=True)
    (extra / "vae").mkdir(parents=True)
    (primary / "vae" / "same.safetensors").write_bytes(b"a")
    (extra / "vae" / "same.safetensors").write_bytes(b"b")

    assert ModelCatalog([primary, extra]).rescan()["vae"] == ["same.safetensors"]


def test_only_the_first_root_is_created_and_written_to(tmp_path: Path) -> None:
    primary, extra = tmp_path / "primary", tmp_path / "extra"
    catalog = ModelCatalog([primary, extra])
    catalog.ensure_dirs()

    assert catalog.root == primary
    assert (primary / "loras").is_dir()
    assert not extra.exists()


def test_scan_skips_dotfile_caches(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    (tmp_path / "loras" / ".cache").mkdir()
    (tmp_path / "loras" / ".cache" / "blob.safetensors").write_bytes(b"x")
    (tmp_path / "loras" / "real.safetensors").write_bytes(b"y")

    assert catalog.rescan()["loras"] == ["real.safetensors"]


def test_annotators_is_scanned(tmp_path: Path) -> None:
    """It holds real weights and the Models panel is meant to show everything on disk."""
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    (tmp_path / "annotators" / "depth.pth").write_bytes(b"x")

    assert catalog.rescan()["annotators"] == ["depth.pth"]


def test_tree_reports_each_root_with_sizes(tmp_path: Path) -> None:
    primary, extra = tmp_path / "primary", tmp_path / "extra"
    (primary / "loras").mkdir(parents=True)
    (extra / "vae").mkdir(parents=True)
    (primary / "loras" / "a.safetensors").write_bytes(b"12345")
    (extra / "vae" / "b.safetensors").write_bytes(b"x")

    roots = ModelCatalog([primary, extra]).tree()

    assert [r["writable"] for r in roots] == [True, False]
    loras = roots[0]["categories"][0]
    assert loras["name"] == "loras" and loras["fileCount"] == 1
    assert loras["children"][0]["sizeBytes"] == 5
    # Categories with nothing in them are left out entirely.
    assert [c["name"] for c in roots[1]["categories"]] == ["vae"]


def test_tree_nests_sharded_model_folders(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    shard = tmp_path / "text_encoders" / "qwen3-4b"
    shard.mkdir(parents=True)
    (shard / "part1.safetensors").write_bytes(b"x")
    (shard / "part2.safetensors").write_bytes(b"y")

    categories = {c["name"]: c for c in catalog.tree()[0]["categories"]}
    folder = categories["text_encoders"]["children"][0]
    assert folder["kind"] == "dir" and folder["fileCount"] == 2
