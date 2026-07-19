from __future__ import annotations

import json
from pathlib import Path

import pytest

from inline_core.extensions.manifest import (
    ManifestError,
    load_manifest,
    package_name,
    parse_manifest,
)


def _manifest(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": 1,
        "id": "acme-tools",
        "name": "Acme Tools",
        "version": "1.0.0",
        "coreCompat": ">=1.2,<2.0",
        "requirements": ["einops>=0.7"],
        "entry": "inline_ext_acme_tools:register",
        "nodes": [{"type": "acme/invert"}],
    }
    base.update(overrides)
    return base


def test_parses_a_valid_manifest() -> None:
    manifest = parse_manifest(_manifest(), expect_id="acme-tools")
    assert manifest.id == "acme-tools"
    assert manifest.package == "inline_ext_acme_tools"
    assert manifest.isolation == "shared"
    assert manifest.node_types() == ("acme/invert",)
    assert manifest.node("acme/invert") is not None
    assert manifest.node("acme/missing") is None


def test_a_node_defaults_to_enabled() -> None:
    node = parse_manifest(_manifest(), expect_id="acme-tools").nodes[0]
    assert node.default_enabled is True
    assert node.models == ()


def test_default_enabled_false_is_respected() -> None:
    manifest = _manifest(nodes=[{"type": "acme/invert", "defaultEnabled": False}])
    assert parse_manifest(manifest, expect_id="acme-tools").nodes[0].default_enabled is False


def test_package_name_maps_dashes_to_underscores() -> None:
    assert package_name("acme-video-tools") == "inline_ext_acme_video_tools"


def test_id_must_match_the_extension_directory() -> None:
    """The directory name and the manifest id are one identity; a mismatch would desynchronize
    the on-disk layout from state.json."""
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_manifest(), expect_id="something-else")
    assert "must equal the extension directory name" in str(excinfo.value)


def test_entry_must_live_in_the_extensions_own_package() -> None:
    """This invariant makes module ownership unambiguous - without it an extension could ship a
    top-level package that shadows another extension's code."""
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_manifest(entry="numpy.core:register"), expect_id="acme-tools")
    assert "must live inside the extension's package" in str(excinfo.value)


def test_entry_is_required() -> None:
    bad = _manifest()
    del bad["entry"]
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(bad, expect_id="acme-tools")
    assert "$.entry" in str(excinfo.value)


def test_reports_every_problem_at_once() -> None:
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest({"schema": 99, "id": "BAD ID"}, expect_id="BAD ID")
    problems = excinfo.value.problems
    assert len(problems) >= 3
    assert any("$.schema" in p for p in problems)
    assert any("$.id" in p for p in problems)
    assert any("$.nodes" in p for p in problems)


def test_rejects_duplicate_node_types() -> None:
    bad = _manifest(nodes=[{"type": "acme/x"}, {"type": "acme/x"}])
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(bad, expect_id="acme-tools")
    assert "declared twice" in str(excinfo.value)


def test_rejects_a_malformed_node_type() -> None:
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_manifest(nodes=[{"type": "notnamespaced"}]), expect_id="acme-tools")
    assert "owner/name" in str(excinfo.value)


def test_rejects_an_empty_node_list() -> None:
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_manifest(nodes=[]), expect_id="acme-tools")
    assert "$.nodes" in str(excinfo.value)


def test_rejects_ui_paths_escaping_the_repository() -> None:
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_manifest(ui="../../etc/passwd"), expect_id="acme-tools")
    assert "relative path inside the repository" in str(excinfo.value)


def test_rejects_model_filename_with_a_path_separator() -> None:
    bad = _manifest(
        nodes=[
            {
                "type": "acme/invert",
                "models": [
                    {
                        "id": "w",
                        "label": "W",
                        "category": "checkpoints",
                        "repo": "acme/w",
                        "repoFile": "w.safetensors",
                        "filename": "../../w.safetensors",
                    }
                ],
            }
        ]
    )
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(bad, expect_id="acme-tools")
    assert "bare filename" in str(excinfo.value)


def test_rejects_an_unknown_model_category() -> None:
    """The category decides where the file lands AND which options_from dropdown lists it, so a
    wrong one downloads successfully and then reports the model as missing."""
    bad = _manifest(
        nodes=[
            {
                "type": "acme/invert",
                "models": [
                    {
                        "id": "w",
                        "label": "W",
                        "category": "my_custom_models",
                        "repo": "acme/w",
                        "repoFile": "w.safetensors",
                        "filename": "w.safetensors",
                    }
                ],
            }
        ]
    )
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(bad, expect_id="acme-tools")
    assert "must be one of" in str(excinfo.value)
    assert "diffusion_models" in str(excinfo.value)


def test_prebuilt_requires_https_and_a_real_digest() -> None:
    bad = _manifest(
        prebuilt=[
            {"platform": "linux-x86_64", "python": "cp311", "url": "http://x/y", "sha256": "nope"}
        ]
    )
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(bad, expect_id="acme-tools")
    assert "must be https" in str(excinfo.value)
    assert "64-character hex" in str(excinfo.value)


def test_load_manifest_reads_from_disk(tmp_path: Path) -> None:
    (tmp_path / "inline-extension.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    assert load_manifest(tmp_path, expect_id="acme-tools").name == "Acme Tools"


def test_load_manifest_missing_file_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(tmp_path, expect_id="acme-tools")
    assert "inline-extension.json" in str(excinfo.value)
