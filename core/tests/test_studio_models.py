"""The explicit model-download backend (studio/models.py) - the torch-free bits: which repo file a
component pulls and the requirements view for unknown node types. The Hugging Face download itself
is not exercised here (no network); the runner/requirements tests cover presence logic."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from inline_core.studio.models import (
    ModelDownloads,
    _component_json,
    _progress_tqdm,
)


def test_progress_tqdm_forwards_the_download_fraction():
    """The tqdm subclass handed to hf_hub_download must turn its byte counter into a 0..0.99
    fraction on our callback."""
    pytest.importorskip("tqdm")
    seen: list[float] = []
    cls = _progress_tqdm(lambda frac, status: seen.append(frac), "VAE")
    bar = cls(total=100, disable=True)
    bar.update(50)
    bar.update(50)  # reaches total, but the cap keeps 1.0 for the post-move "ready"
    assert seen == [0.5, 0.99]


def test_requirements_empty_for_unknown_node_type():
    downloads = ModelDownloads(events=None)
    assert downloads.requirements("no/such-node") == {
        "components": [],
        "allPresent": True,
        "estimate": None,  # no requirements + no policy -> no fit estimate
    }


def test_download_retries_anonymously_when_a_cached_token_is_invalid(tmp_path, monkeypatch):
    """A stale/invalid cached HF token 401s even on a public repo (masked as "not found"). The
    download must drop the token and retry anonymously, not fail."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import HfHubHTTPError

    tokens_seen: list[object] = []

    def fake_download(repo, rfilename, local_dir, token=None, tqdm_class=None):
        tokens_seen.append(token)
        if token is not False:  # ambient (stale) token -> 401 the way HF does
            response = types.SimpleNamespace(status_code=401, headers={}, request=None)
            raise HfHubHTTPError("401 Client Error. Repository Not Found", response=response)
        dest = Path(local_dir) / rfilename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        return str(dest)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)

    downloads = ModelDownloads(events=None)
    comp = types.SimpleNamespace(repo="lokCX/4x-Ultrasharp", repo_file="w.pth", filename="w.pth",
                                 label="4x UltraSharp")
    provider = types.SimpleNamespace(download_target=lambda c: tmp_path)
    downloads._download_component(provider, comp, lambda frac, status: None)

    assert (tmp_path / "w.pth").read_bytes() == b"data"
    assert tokens_seen == [None, False]  # tried the ambient token, then fell back to anonymous


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
