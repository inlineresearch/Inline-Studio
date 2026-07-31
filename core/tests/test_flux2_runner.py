"""The FLUX.2 node's descriptor and the decisions it makes before any weights load.

Import-guarded and GPU-free: everything here is the logic that turns a picked checkpoint plus the
user's params into a pipeline call.
"""

from __future__ import annotations

import pytest

from inline_core.models.flux2 import variants as V

runner = pytest.importorskip("inline_core.models.flux2.runner")


def test_one_node_covers_the_whole_family() -> None:
    assert runner.FLUX2.type == "black-forest-labs/flux-2"
    assert runner.FLUX2.output_kind is not None, "it produces a Frame with take history"
    # Every variant is reachable from the one node's dropdown, and only real ones are listed: the
    # variant is normally identified from the checkpoint and served as the default, so an "auto"
    # entry would only hide which build actually ran.
    variant = next(p for p in runner.FLUX2.params if p.key == "variant")
    assert [o.value for o in variant.options] == [v.key for v in V.VARIANTS]


def test_references_are_a_list_port_and_there_is_no_img2img_strength() -> None:
    from inline_core.graph.schema import PortKind

    image = runner.FLUX2.input("image")
    assert image is not None and image.kind is PortKind.IMAGE_LIST
    # FLUX.2 conditions on references through the whole denoise, so there is nothing to partially
    # denoise from - a stray strength param would be a lie.
    assert not any(p.key == "strength" for p in runner.FLUX2.params)


def test_a_structure_map_can_be_wired() -> None:
    from inline_core.graph.schema import PortKind, port_satisfies

    control = runner.FLUX2.input("control_image")
    assert control is not None and control.kind is PortKind.CONTROL
    # Apply ControlNet and Control Space both emit an image-kind output.
    assert port_satisfies(PortKind.IMAGE, PortKind.CONTROL)


@pytest.mark.parametrize(
    ("variant_key", "steps", "guidance"),
    [("klein-4b", 4, 1.0), ("klein-4b-base", 50, 4.0), ("dev", 28, 4.0)],
)
def test_auto_sampler_defaults_come_from_the_checkpoint(
    variant_key: str, steps: int, guidance: float
) -> None:
    # The descriptor ships sentinels, so switching checkpoints moves the schedule without the user
    # touching the settings panel.
    defaults = runner.FLUX2.defaults()
    assert defaults["steps"] == 0 and defaults["guidance"] == -1.0
    variant = V.get(variant_key)
    assert runner._resolve_steps(defaults, variant) == steps
    assert runner._resolve_guidance(defaults, variant) == guidance


def test_an_explicit_setting_always_wins() -> None:
    variant = V.get("klein-4b")
    assert runner._resolve_steps({"steps": 12}, variant) == 12
    assert runner._resolve_guidance({"guidance": 0.0}, variant) == 0.0


def test_negative_prompt_is_dropped_where_it_does_nothing() -> None:
    params = {"negative_prompt": "blurry"}
    # Only an undistilled klein runs real CFG.
    assert runner._resolve_negative(params, V.get("klein-4b-base")) == "blurry"
    assert runner._resolve_negative(params, V.get("klein-4b")) is None
    assert runner._resolve_negative(params, V.get("dev")) is None
    assert runner._resolve_negative({"negative_prompt": "  "}, V.get("klein-4b-base")) is None


def test_dimensions_snap_to_the_latent_grid() -> None:
    # 2x2 latent packing on top of an 8x VAE means both sides must be multiples of 16.
    assert runner._snap(1024) == 1024
    assert runner._snap(1020) == 1008
    assert runner._snap(10) == 64, "never below the model's 64px floor"


def test_the_kv_variant_takes_no_guidance_argument() -> None:
    # Flux2KleinKVPipeline.__call__ has no guidance_scale at all, so passing one would TypeError.
    import inspect

    diffusers = pytest.importorskip("diffusers")
    signature = inspect.signature(diffusers.Flux2KleinKVPipeline.__call__)
    assert "guidance_scale" not in signature.parameters
    assert V.get("klein-9b-kv").pipeline == "klein-kv"


def test_the_descriptor_matches_the_pipelines_it_drives() -> None:
    """A drifting diffusers signature should fail here, not mid-denoise."""
    import inspect

    diffusers = pytest.importorskip("diffusers")
    for cls in (diffusers.Flux2KleinPipeline, diffusers.Flux2Pipeline):
        params = inspect.signature(cls.__call__).parameters
        for name in ("image", "prompt_embeds", "max_sequence_length", "text_encoder_out_layers"):
            assert name in params, f"{cls.__name__} lost {name}"
    # Only the klein pipeline has a negative path, which is what supports_negative_prompt encodes.
    klein = inspect.signature(diffusers.Flux2KleinPipeline.__call__).parameters
    dev = inspect.signature(diffusers.Flux2Pipeline.__call__).parameters
    assert "negative_prompt_embeds" in klein
    assert "negative_prompt_embeds" not in dev


def test_a_variant_override_that_contradicts_the_checkpoint_is_ignored(tmp_path) -> None:
    """The dropdown defaults to a concrete variant and the settings panel persists the whole draft,
    so a node can end up carrying a variant that no longer matches its file. Honouring it would
    build the wrong pipeline and produce noise; the file wins."""
    from tests.test_flux2_resolve import _write_header_only
    from tests.test_flux2_variants import DEV, KLEIN_4B, _shapes

    dev_file = _write_header_only(tmp_path / "flux2_dev.safetensors", _shapes(DEV))
    # A stale pick from when this node pointed at a klein checkpoint.
    variant, config = runner._identify(str(dev_file), {"variant": "klein-4b"})
    assert variant is V.get("dev"), "the checkpoint decides, not the stale override"
    assert config["joint_attention_dim"] == 15360

    # A pick that agrees with the file is still honoured: it is the only way to say "this is a Base
    # build" for a file whose name does not say so, which shapes cannot reveal.
    klein_file = _write_header_only(tmp_path / "anon.safetensors", _shapes(KLEIN_4B))
    forced, _ = runner._identify(str(klein_file), {"variant": "klein-4b-base"})
    assert forced is V.get("klein-4b-base")
