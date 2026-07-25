"""The Krea 2 reference -> diffusers key map, checked against the real transformer's own shapes.

``Krea2Transformer2DModel`` has no ``from_single_file``, so this rename is the only thing standing
between the ComfyUI checkpoint and a working pipeline. A meta-device model costs no memory, so the
test asserts the full 430-key round trip rather than a sample.
"""

from __future__ import annotations

import pytest

from inline_core.errors import ComponentError
from inline_core.models.krea2 import convert

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")


def _reference_key(diffusers_key: str) -> str:
    """The reference name for a diffusers one - the inverse of the rename, for building fixtures."""
    key = diffusers_key
    for old, new in (
        ("transformer_blocks.", "blocks."),
        ("text_fusion.", "txtfusion."),
        ("img_in.", "first."),
        ("time_embed.linear_1.", "tmlp.0."),
        ("time_embed.linear_2.", "tmlp.2."),
        ("time_mod_proj.", "tproj.1."),
        ("txt_in.norm.weight", "txtmlp.0.scale"),
        ("txt_in.linear_1.", "txtmlp.1."),
        ("txt_in.linear_2.", "txtmlp.3."),
        ("final_layer.linear.", "last.linear."),
        ("final_layer.norm.weight", "last.norm.scale"),
        ("final_layer.scale_shift_table", "last.modulation.lin"),
    ):
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    for old, new in (
        (".attn.norm_q.weight", ".attn.qknorm.qnorm.scale"),
        (".attn.norm_k.weight", ".attn.qknorm.knorm.scale"),
        (".attn.to_q.", ".attn.wq."),
        (".attn.to_k.", ".attn.wk."),
        (".attn.to_v.", ".attn.wv."),
        (".attn.to_out.0.", ".attn.wo."),
        (".attn.to_gate.", ".attn.gate."),
        (".ff.up.", ".mlp.up."),
        (".ff.gate.", ".mlp.gate."),
        (".ff.down.", ".mlp.down."),
        (".norm1.weight", ".prenorm.scale"),
        (".norm2.weight", ".postnorm.scale"),
    ):
        key = key.replace(old, new)
    if key.endswith(".scale_shift_table"):
        key = key[: -len(".scale_shift_table")] + ".mod.lin"
    return key


@pytest.fixture(scope="module")
def shapes() -> dict[str, tuple[int, ...]]:
    """The real Krea 2 transformer's state-dict shapes, built on meta (no memory, no weights)."""
    from diffusers import Krea2Transformer2DModel

    with torch.device("meta"):
        model = Krea2Transformer2DModel()
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _reference_state(shapes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    """A synthetic reference-layout checkpoint: every diffusers tensor under its reference name, in
    the reference's flat shape for the modulation tables."""
    state: dict[str, object] = {}
    for name, shape in shapes.items():
        flat = name.endswith(".scale_shift_table") and "transformer_blocks." in name
        source = (int(torch.tensor(shape).prod()),) if flat else shape
        state[_reference_key(name)] = torch.zeros(source, device="meta")
    return state


def test_the_reference_layout_maps_onto_every_diffusers_tensor(shapes) -> None:
    state = _reference_state(shapes)
    # The real checkpoint has 430 tensors; a drift in either layout shows up here first.
    assert len(state) == len(shapes) == 430

    converted = convert.convert_state_dict(state, shapes)

    assert set(converted) == set(shapes)
    for name, tensor in converted.items():
        assert tuple(tensor.shape) == shapes[name], name


def test_flat_modulation_weights_are_reshaped_to_the_scale_shift_table(shapes) -> None:
    # The one genuine layout difference: reference stores 6*6144 flat, diffusers wants (6, 6144).
    assert shapes["transformer_blocks.0.scale_shift_table"] == (6, 6144)
    state = {"blocks.0.mod.lin": torch.zeros(36864, device="meta")}

    converted = convert.convert_state_dict(
        state, {"transformer_blocks.0.scale_shift_table": (6, 6144)}
    )

    assert tuple(converted["transformer_blocks.0.scale_shift_table"].shape) == (6, 6144)


def test_an_already_diffusers_named_checkpoint_passes_through(shapes) -> None:
    state = {name: torch.zeros(shape, device="meta") for name, shape in shapes.items()}

    converted = convert.convert_state_dict(state, shapes)

    assert set(converted) == set(shapes)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("blocks.0.attn.wq", "transformer_blocks.0.attn.to_q"),
        ("blocks.27.attn.wo", "transformer_blocks.27.attn.to_out.0"),
        ("blocks.3.attn.gate", "transformer_blocks.3.attn.to_gate"),
        ("blocks.5.mlp.down", "transformer_blocks.5.ff.down"),
        ("txtfusion.layerwise_blocks.1.attn.wv", "text_fusion.layerwise_blocks.1.attn.to_v"),
        ("txtfusion.refiner_blocks.0.mlp.up", "text_fusion.refiner_blocks.0.ff.up"),
        ("txtfusion.projector", "text_fusion.projector"),
        ("first", "img_in"),
        ("last.linear", "final_layer.linear"),
        ("tmlp.0", "time_embed.linear_1"),
        ("tproj.1", "time_mod_proj"),
        ("txtmlp.1", "txt_in.linear_1"),
    ],
)
def test_module_alias_maps_reference_lora_paths(stem: str, expected: str, shapes) -> None:
    # ostris' training adapter is named this way; the official Comfy-Org LoRAs are not.
    assert convert.module_alias(stem) == expected
    assert f"{expected}.weight" in shapes


def test_module_alias_is_none_for_an_already_diffusers_path() -> None:
    assert convert.module_alias("transformer_blocks.0.attn.to_q") is None


def test_a_comfy_quantized_build_is_refused_by_name(shapes) -> None:
    state = _reference_state(shapes)
    state["blocks.0.attn.wq.weight_scale"] = torch.zeros(1, device="meta")

    with pytest.raises(ComponentError, match="quantized"):
        convert.convert_state_dict(state, shapes)


def test_an_unrelated_checkpoint_is_refused_rather_than_partially_loaded(shapes) -> None:
    with pytest.raises(ComponentError, match="unrecognised"):
        convert.convert_state_dict({"model.diffusion_model.foo": torch.zeros(1)}, shapes)


def test_a_truncated_checkpoint_is_refused(shapes) -> None:
    state = _reference_state(shapes)
    del state["blocks.0.attn.wq.weight"]

    with pytest.raises(ComponentError, match="missing"):
        convert.convert_state_dict(state, shapes)
