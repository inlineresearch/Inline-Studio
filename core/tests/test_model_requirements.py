"""The requirements-provider registry that replaced the hardcoded Z-Image node-type check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inline_core.extensions.manifest import ModelRequirement
from inline_core.extensions.models import ManifestRequirements
from inline_core.models.requirements import ModelComponent, RequirementsRegistry
from inline_core.studio.models import ModelDownloads


class _StubProvider:
    def __init__(self, components: list[ModelComponent]) -> None:
        self._components = components

    def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
        return self._components

    def download_target(self, component: ModelComponent) -> Path:
        return Path("models") / component.category

    def estimate(self, policy: Any) -> dict[str, Any] | None:
        return {"plan": "resident", "fits": True}


def _component(present: bool = False) -> ModelComponent:
    return ModelComponent(
        id="w",
        label="Weights",
        category="checkpoints",
        present=present,
        filename="w.safetensors",
        repo="acme/w",
        repo_file="w.safetensors",
    )


def test_optional_component_is_a_suggestion_not_a_missing_requirement() -> None:
    """A suggested (optional) component that isn't on disk must not flip ``allPresent`` to False,
    and "Download all" must skip it - control is opt-in, so it never blocks a plain run."""
    optional = ModelComponent(
        id="controlnet", label="ControlNet", category="controlnet", present=False,
        filename="cn.safetensors", repo="acme/cn", repo_file="cn.safetensors", optional=True,
    )
    registry = RequirementsRegistry()
    registry.register("acme/thing", _StubProvider([_component(present=True), optional]))
    downloads = ModelDownloads(events=None, requirements=registry)

    payload = downloads.requirements("acme/thing")
    assert payload["allPresent"] is True  # required present; the optional one is ignored
    cn = next(c for c in payload["components"] if c["id"] == "controlnet")
    assert cn["optional"] is True and cn["present"] is False


def test_control_space_provider_offers_the_controlnet_suggestion() -> None:
    from inline_core.models.controlspace import ControlSpaceProvider

    components = ControlSpaceProvider().components()
    assert [c.id for c in components] == ["controlnet"]
    assert components[0].optional is True


def test_unregistered_node_type_reports_no_requirements() -> None:
    """The pre-refactor behaviour for any non-Z-Image node, now the general case."""
    downloads = ModelDownloads(events=None, requirements=RequirementsRegistry())
    assert downloads.requirements("acme/unknown") == {
        "components": [],
        "allPresent": True,
        "estimate": None,
    }


def test_provider_drives_the_popup_payload() -> None:
    registry = RequirementsRegistry()
    registry.register("acme/thing", _StubProvider([_component(present=False)]))
    downloads = ModelDownloads(events=None, policy=object(), requirements=registry)

    payload = downloads.requirements("acme/thing")

    assert payload["allPresent"] is False
    assert payload["estimate"] == {"plan": "resident", "fits": True}
    component = payload["components"][0]
    assert component["id"] == "w"
    assert component["localPath"] == "checkpoints/w.safetensors"
    assert component["source"] == "acme/w/w.safetensors"


def test_estimate_is_omitted_without_a_policy() -> None:
    registry = RequirementsRegistry()
    registry.register("acme/thing", _StubProvider([_component()]))
    downloads = ModelDownloads(events=None, policy=None, requirements=registry)
    assert downloads.requirements("acme/thing")["estimate"] is None


def test_a_raising_provider_never_breaks_the_popup() -> None:
    """Providers are extension code: a bad one degrades to "no requirements", it does not take
    down the model popup for every node."""

    class Exploding:
        def components(self, params: dict[str, object] | None = None) -> list[ModelComponent]:
            raise RuntimeError("boom")

        def download_target(self, component: ModelComponent) -> Path:
            raise RuntimeError("boom")

        def estimate(self, policy: Any) -> dict[str, Any] | None:
            raise RuntimeError("boom")

    registry = RequirementsRegistry()
    registry.register("acme/bad", Exploding())
    downloads = ModelDownloads(events=None, policy=object(), requirements=registry)

    payload = downloads.requirements("acme/bad")
    assert payload["components"] == []
    assert payload["estimate"] is None


def test_registry_unregister_removes_the_provider() -> None:
    registry = RequirementsRegistry()
    registry.register("acme/thing", _StubProvider([_component()]))
    assert registry.has("acme/thing")
    registry.unregister("acme/thing")
    assert not registry.has("acme/thing")
    assert registry.get("acme/thing") is None


def test_zimage_provider_matches_the_module_level_requirements() -> None:
    """The migration must be behaviour-preserving: the provider is a thin wrapper, not a rewrite."""
    from inline_core.models.zimage.provider import ZImageProvider
    from inline_core.models.zimage.requirements import zimage_requirements

    assert ZImageProvider().components() == zimage_requirements()


def test_manifest_requirements_refuse_an_unknown_category() -> None:
    """Manifest validation rejects these first; reaching the provider means the install directory
    was hand-edited. Raising beats guessing a substitute, which would silently download weights
    into a folder the node's options_from dropdown never reads."""
    import pytest

    escaping = ModelRequirement(
        id="w",
        label="W",
        category="../../../etc",
        repo="acme/w",
        repo_file="w.safetensors",
        filename="w.safetensors",
    )
    with pytest.raises(ValueError, match="unknown model category"):
        ManifestRequirements((escaping,)).components()


def test_manifest_requirements_keep_a_valid_category() -> None:
    valid = ModelRequirement(
        id="w",
        label="W",
        category="upscale_models",
        repo="acme/w",
        repo_file="nested/w.safetensors",
        filename="",
    )
    component = ManifestRequirements((valid,)).components()[0]
    assert component.category == "upscale_models"
    assert component.filename == "w.safetensors", "filename falls back to the repo file's basename"


def test_manifest_requirements_have_no_fit_estimate() -> None:
    """A wrong "this will fit" is worse than none, and the manifest carries no footprint data."""
    provider = ManifestRequirements(())
    assert provider.estimate(object()) is None
