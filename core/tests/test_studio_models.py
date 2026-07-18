"""The explicit model-download backend (studio/models.py) - the torch-free bits: which repo file a
component pulls and the requirements view for unknown node types. The Hugging Face download itself
is not exercised here (no network); the runner/requirements tests cover presence logic."""

from __future__ import annotations

import types
from pathlib import Path

from inline_core.studio.models import (
    ModelDownloads,
    _component_json,
    _dir_size,
    _wanted_files,
)


def _fake_api(files: list[tuple[str, int]]):
    siblings = [types.SimpleNamespace(rfilename=name, size=size) for name, size in files]
    info = types.SimpleNamespace(siblings=siblings)
    return types.SimpleNamespace(model_info=lambda repo, files_metadata=False: info)


def test_wanted_files_picks_the_one_repo_file_with_size():
    api = _fake_api(
        [
            ("split_files/vae/ae.safetensors", 100),
            ("split_files/diffusion_models/z_image_bf16.safetensors", 999),
            ("README.md", 2),
        ]
    )
    comp = types.SimpleNamespace(repo="r", repo_file="split_files/vae/ae.safetensors")
    assert _wanted_files(api, comp) == [("split_files/vae/ae.safetensors", 100)]


def test_wanted_files_falls_back_to_zero_size_when_absent():
    # Not in the listing -> still attempt the download (size 0 = unknown total, no live fraction).
    api = _fake_api([("split_files/vae/ae.safetensors", 100)])
    comp = types.SimpleNamespace(repo="r", repo_file="split_files/vae/missing.safetensors")
    assert _wanted_files(api, comp) == [("split_files/vae/missing.safetensors", 0)]


def test_dir_size_sums_nested_files(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    nested = tmp_path / ".cache" / "download"
    nested.mkdir(parents=True)
    (nested / "b.incomplete").write_bytes(b"y" * 25)
    assert _dir_size(tmp_path) == 35


def test_dir_size_missing_dir_is_zero(tmp_path: Path):
    assert _dir_size(tmp_path / "nope") == 0


def test_requirements_empty_for_unknown_node_type():
    downloads = ModelDownloads(events=None)
    assert downloads.requirements("no/such-node") == {
        "components": [],
        "allPresent": True,
        "estimate": None,  # no requirements + no policy -> no fit estimate
    }


def test_component_json_shape():
    comp = types.SimpleNamespace(
        id="vae",
        label="VAE",
        category="vae",
        present=False,
        local_path="vae/ae.safetensors",
        repo="Comfy-Org/z_image",
        filename="ae.safetensors",
        repo_file="split_files/vae/ae.safetensors",
    )
    assert _component_json(comp) == {
        "id": "vae",
        "label": "VAE",
        "category": "vae",
        "present": False,
        "localPath": "vae/ae.safetensors",
        "repo": "Comfy-Org/z_image",
        # "which model" is the repo + the exact file this component pulls.
        "source": "Comfy-Org/z_image/ae.safetensors",
    }
