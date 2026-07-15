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
