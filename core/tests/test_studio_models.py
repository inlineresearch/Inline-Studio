"""The explicit model-download backend (studio/models.py) — the torch-free bits: file selection,
subfolder flattening, and the requirements view for unknown node types. The actual Hugging Face
download is not exercised here (no network); the runner/requirements tests cover presence logic."""

from __future__ import annotations

import types

from inline_core.studio.models import (
    ModelDownloads,
    _component_json,
    _flatten_rel,
    _wanted_files,
)


def test_flatten_rel_strips_subfolder_prefix():
    # Subfolder components flatten into the target dir (strip the leading vae/ etc.).
    assert _flatten_rel("vae/config.json", ("vae",)) == "config.json"
    assert _flatten_rel("text_encoder/model.safetensors", ("text_encoder", "tokenizer")) == (
        "model.safetensors"
    )
    assert _flatten_rel("tokenizer/tokenizer.json", ("text_encoder", "tokenizer")) == (
        "tokenizer.json"
    )


def test_flatten_rel_keeps_layout_for_whole_repo():
    # A whole-repo (pipeline) download has no subfolders and keeps its diffusers layout.
    assert _flatten_rel("model_index.json", ()) == "model_index.json"
    assert _flatten_rel("transformer/diffusion_pytorch_model.safetensors", ()) == (
        "transformer/diffusion_pytorch_model.safetensors"
    )


def _fake_api(files: list[tuple[str, int]]):
    siblings = [types.SimpleNamespace(rfilename=name, size=size) for name, size in files]
    info = types.SimpleNamespace(siblings=siblings)
    return types.SimpleNamespace(model_info=lambda repo, files_metadata=False: info)


def test_wanted_files_filters_to_subfolders():
    api = _fake_api(
        [
            ("vae/config.json", 10),
            ("vae/diffusion_pytorch_model.safetensors", 100),
            ("transformer/x.safetensors", 999),
            ("model_index.json", 5),
        ]
    )
    comp = types.SimpleNamespace(repo="r", subfolders=("vae",))
    assert _wanted_files(api, comp) == [
        ("vae/config.json", 10),
        ("vae/diffusion_pytorch_model.safetensors", 100),
    ]


def test_wanted_files_whole_repo_skips_boilerplate():
    api = _fake_api(
        [
            (".gitattributes", 1),
            ("README.md", 2),
            ("model_index.json", 5),
            ("transformer/x.safetensors", 999),
        ]
    )
    comp = types.SimpleNamespace(repo="r", subfolders=())
    got = dict(_wanted_files(api, comp))
    assert ".gitattributes" not in got and "README.md" not in got
    assert got == {"model_index.json": 5, "transformer/x.safetensors": 999}


def test_requirements_empty_for_unknown_node_type():
    downloads = ModelDownloads(events=None)
    assert downloads.requirements("no/such-node") == {"components": [], "allPresent": True}


def test_component_json_shape():
    comp = types.SimpleNamespace(
        id="vae",
        label="VAE",
        category="vae",
        present=False,
        local_path="vae/z-image-turbo",
        repo="Tongyi-MAI/Z-Image-Turbo",
    )
    assert _component_json(comp) == {
        "id": "vae",
        "label": "VAE",
        "category": "vae",
        "present": False,
        "localPath": "vae/z-image-turbo",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
    }
