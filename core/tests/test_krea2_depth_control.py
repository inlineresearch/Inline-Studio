"""Krea 2 depth control: the surgery that ports the public depth control-LoRA onto the diffusers
transformer. The offline-checkable contracts are the key remap (so ``load_state_dict`` has no
unexpected keys on the real 450-tensor checkpoint) and the input-projection expansion shape.
"""

from __future__ import annotations

import pytest

from inline_core.models.krea2.convert import convert_key

# reference (krea-2) block-linear name -> diffusers name, the eight modules the adapter wraps.
_TARGETS = {
    "attn.wq": "attn.to_q",
    "attn.wk": "attn.to_k",
    "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0",
    "attn.gate": "attn.to_gate",
    "mlp.gate": "ff.gate",
    "mlp.up": "ff.up",
    "mlp.down": "ff.down",
}
_LAYERS = 28  # Krea 2 has 28 main transformer blocks


def _reference_keys() -> list[str]:
    """Every tensor name in the real checkpoint: 28 blocks x 8 targets x {A,B} + the input proj."""
    keys = ["first.weight", "first.bias"]
    for i in range(_LAYERS):
        for name in _TARGETS:
            keys += [f"blocks.{i}.{name}.A", f"blocks.{i}.{name}.B"]
    return keys


def _expected_diffusers_keys() -> set[str]:
    keys = {"img_in.weight", "img_in.bias"}
    for i in range(_LAYERS):
        for name in _TARGETS.values():
            keys |= {f"transformer_blocks.{i}.{name}.A", f"transformer_blocks.{i}.{name}.B"}
    return keys


def test_every_depth_lora_key_maps_onto_the_diffusers_transformer() -> None:
    """The whole checkpoint remaps cleanly - no key would land as ``unexpected`` at load time."""
    reference = _reference_keys()
    assert len(reference) == _LAYERS * len(_TARGETS) * 2 + 2 == 450
    expected = _expected_diffusers_keys()
    assert {convert_key(k) for k in reference} == expected


def test_module_paths_excludes_input_projection() -> None:
    dc = pytest.importorskip("inline_core.models.krea2.depth_control")
    state = {
        "img_in.weight": 0,
        "img_in.bias": 0,
        "transformer_blocks.0.attn.to_q.A": 0,
        "transformer_blocks.0.attn.to_q.B": 0,
        "transformer_blocks.0.attn.to_out.0.A": 0,
        "transformer_blocks.0.ff.up.B": 0,
    }
    assert dc._module_paths(state) == [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.ff.up",
    ]


def test_control_input_layer_doubles_width_and_broadcasts_over_batch() -> None:
    dc = pytest.importorskip("inline_core.models.krea2.depth_control")
    import torch
    import torch.nn as nn

    layer = dc.ControlInputLayer(nn.Linear(64, 6144))
    assert tuple(layer.weight.shape) == (6144, 128)  # 64 noise + 64 depth

    layer.ctrl = torch.zeros(1, 5, 64)  # depth latent (batch 1)
    out = layer(torch.zeros(2, 5, 64))  # CFG-doubled noisy latent -> ctrl broadcast to batch 2
    assert tuple(out.shape) == (2, 5, 6144)


def test_control_input_layer_zeros_when_no_depth_set() -> None:
    dc = pytest.importorskip("inline_core.models.krea2.depth_control")
    import torch
    import torch.nn as nn

    layer = dc.ControlInputLayer(nn.Linear(64, 6144))
    assert layer.ctrl is None
    out = layer(torch.zeros(1, 5, 64))  # no depth -> concat zeros, still a valid forward
    assert tuple(out.shape) == (1, 5, 6144)


def test_set_control_strength_updates_every_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    dc = pytest.importorskip("inline_core.models.krea2.depth_control")
    import torch.nn as nn

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = dc.LoRALinear(nn.Linear(8, 8), rank=4)
            self.b = dc.LoRALinear(nn.Linear(8, 8), rank=4)

    model = _Tiny()
    dc.set_control_strength(model, 0.5)
    assert model.a.scale == 0.5 and model.b.scale == 0.5
