"""The H3 loader, round-tripped at a size that fits in memory.

A synthetic checkpoint is written in the *reference* layout by inverting the key plan, then loaded
back through the real streaming path. If the split, the half-swap or the de-interleave ran the wrong
way, the recovered weights differ from the originals and this fails - which is the whole point,
because at full size those three mistakes produce a video that plays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from inline_core.errors import ComponentError  # noqa: E402
from inline_core.models.keymap import (  # noqa: E402
    AssertEqual,
    Rename,
    RowLayout,
    Split,
    SwapHalves,
)
from inline_core.models.minimaxh3 import keys as h3keys  # noqa: E402
from inline_core.models.minimaxh3.load import (  # noqa: E402
    detect_source_layout,
    expected_inv_freq,
    load_transformer,
    transformer_kwargs,
)
from inline_core.models.minimaxh3.vendor import MiniMaxH3Transformer3DModel  # noqa: E402

#: Small enough to build for real, same shape of problem as the released geometry.
TINY = dict(
    num_attention_heads=3,
    attention_head_dim=8,
    hidden_size=16,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=8,
    in_channels=4,
    audio_in_channels=4,
    text_dim=8,
    freq_dim=8,
    time_embed_hidden_dim=16,
    time_embed_dim=8,
    rope_freq_dim=4,
)
TINY_PLAN = dict(num_blocks=1, num_refiner_blocks=1, head_dim=TINY["attention_head_dim"])


def _tiny_model() -> Any:
    torch.manual_seed(0)
    model = MiniMaxH3Transformer3DModel(**TINY)
    return model.eval()


def _invert(plan: Any, state: dict[str, torch.Tensor], layout: RowLayout) -> dict[str, Any]:
    """Write the reference-format checkpoint a given diffusers state dict came from."""
    heads = TINY["num_attention_heads"]
    head_dim = TINY["attention_head_dim"]
    source: dict[str, torch.Tensor] = {}
    for key, action in plan.actions.items():
        if isinstance(action, Rename):
            source[key] = state[action.target].clone()
        elif isinstance(action, SwapHalves):
            fused = state[action.target]
            half = fused.shape[0] // 2
            source[key] = torch.cat([fused[half:], fused[:half]]).clone()
        elif isinstance(action, Split):
            parts = [state[t] for t in action.targets]
            stacked = torch.cat(parts)
            if layout is RowLayout.INTERLEAVED:
                rows = []
                for head in range(heads):
                    for part in parts:
                        rows.append(part[head * head_dim : (head + 1) * head_dim])
                stacked = torch.cat(rows)
            source[key] = stacked.clone()
        elif isinstance(action, AssertEqual):
            source[key] = expected_inv_freq(
                TINY["rope_freq_dim"], 10000.0
            ).to(torch.float32)
    return source


#: The same geometry in the source config's naming, which is what a publisher ships beside a
#: checkpoint. Without it the loader falls back to the released defaults, which is correct for the
#: published single-file builds and wrong for this one.
TINY_CONFIG = {
    "num_attention_heads": TINY["num_attention_heads"],
    "attention_head_dim": TINY["attention_head_dim"],
    "hidden_size": TINY["hidden_size"],
    "num_layers": TINY["num_layers"],
    "token_refiner_num_layers": TINY["num_refiner_layers"],
    "ffn_hidden_size": TINY["ffn_dim"],
    "latents_dim": TINY["in_channels"],
    "audio_latents_dim": TINY["audio_in_channels"],
    "text_dim": TINY["text_dim"],
    "timestep_input_dim": TINY["freq_dim"],
    "time_embed_hidden_size": TINY["time_embed_hidden_dim"],
    "time_embed_dim": TINY["time_embed_dim"],
    "rope_inv_freq_len": TINY["rope_freq_dim"],
}


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    import json

    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    (path.parent / "config.json").write_text(json.dumps(TINY_CONFIG))
    return path


@pytest.fixture()
def reference(tmp_path: Path):  # type: ignore[no-untyped-def]
    def build(layout: RowLayout) -> tuple[Path, dict[str, torch.Tensor]]:
        model = _tiny_model()
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        plan = h3keys.build_plan(_name(layout), **TINY_PLAN)
        path = _write(tmp_path / f"h3_{layout.value}.safetensors", _invert(plan, state, layout))
        return path, state

    return build


def _name(layout: RowLayout) -> str:
    return next(k for k, v in h3keys.SOURCE_LAYOUTS.items() if v is layout)


# --- the round trip -------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [RowLayout.CONTIGUOUS, RowLayout.INTERLEAVED])
def test_a_reference_checkpoint_round_trips_through_the_loader(reference, layout) -> None:  # type: ignore[no-untyped-def]
    path, original = reference(layout)

    loaded = load_transformer(path, dtype=torch.float32)

    recovered = loaded.state_dict()
    assert set(recovered) == set(original)
    for key, want in original.items():
        assert torch.equal(recovered[key], want), key


def test_both_layouts_recover_the_same_weights(reference) -> None:  # type: ignore[no-untyped-def]
    """The two publishers ship the same model, so both files must load to identical weights."""
    contiguous, _ = reference(RowLayout.CONTIGUOUS)
    interleaved, _ = reference(RowLayout.INTERLEAVED)

    a = load_transformer(contiguous, dtype=torch.float32).state_dict()
    b = load_transformer(interleaved, dtype=torch.float32).state_dict()

    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_the_layout_is_measured_from_the_file_not_its_name(reference) -> None:  # type: ignore[no-untyped-def]
    """Renaming a checkpoint must not change how it loads."""
    path, original = reference(RowLayout.INTERLEAVED)
    renamed = path.rename(path.with_name("definitely_a_comfy_org_file.safetensors"))

    recovered = load_transformer(renamed, dtype=torch.float32).state_dict()

    assert torch.equal(recovered["transformer_blocks.0.attn.to_q.weight"],
                       original["transformer_blocks.0.attn.to_q.weight"])


# --- the guards -----------------------------------------------------------------------------------


def test_a_wrong_rope_table_is_refused(reference, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A theta that disagrees with the port drifts geometry across every frame and still renders."""
    path, _ = reference(RowLayout.CONTIGUOUS)
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    tensors["rope.inv_freq"] = expected_inv_freq(TINY["rope_freq_dim"], 500000.0).to(torch.float32)
    broken = _write(tmp_path / "wrong_theta.safetensors", tensors)

    with pytest.raises(ComponentError, match="rope.inv_freq"):
        load_transformer(broken, dtype=torch.float32)


def test_a_checkpoint_that_is_not_h3_is_named_as_such(tmp_path: Path) -> None:
    path = _write(tmp_path / "something_else.safetensors", {"a.weight": torch.zeros(4, 4)})
    with pytest.raises(ComponentError, match="not a MiniMax H3 transformer"):
        load_transformer(path, dtype=torch.float32)


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(ComponentError, match="not found"):
        load_transformer(tmp_path / "absent.safetensors")


def test_an_incomplete_checkpoint_fails_before_any_forward_pass(reference, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, _ = reference(RowLayout.CONTIGUOUS)
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    del tensors["blocks.0.mlp.fc2.weight"]
    truncated = _write(tmp_path / "incomplete.safetensors", tensors)

    # Diagnosed as the plan expecting a tensor the file lacks, which names the missing key rather
    # than surfacing later as a meta-tensor error inside a forward pass.
    with pytest.raises(ComponentError, match="blocks.0.mlp.fc2.weight"):
        load_transformer(truncated, dtype=torch.float32)


# --- config and rope ------------------------------------------------------------------------------


def test_the_source_config_names_map_onto_the_ports_arguments() -> None:
    source = {
        "num_layers": 50, "token_refiner_num_layers": 2, "ffn_hidden_size": 14336,
        "latents_dim": 24, "audio_latents_dim": 32, "timestep_input_dim": 256,
        "time_embed_hidden_size": 5376, "rope_inv_freq_len": 16,
        "adaln_out_features": 96768, "final_adaln_out_features": 10752,
    }
    mapped = transformer_kwargs(source)
    assert mapped["num_refiner_layers"] == 2 and mapped["ffn_dim"] == 14336
    assert mapped["in_channels"] == 24 and mapped["audio_in_channels"] == 32
    assert mapped["freq_dim"] == 256 and mapped["rope_freq_dim"] == 16
    # Derived by the port from the geometry; passing a stale pair would disagree with the weights.
    assert "adaln_out_features" not in mapped
    assert "final_adaln_out_features" not in mapped


def test_the_released_rope_table_solves_to_theta_10000() -> None:
    """The values published in FL2VA/transformer, which the config never states a theta for."""
    # Literals are the shipped table rounded to 8 decimals, which is what bounds the tolerance
    # here. The exact check runs at load time against the file itself, not against these.
    published = torch.tensor(
        [1.0, 0.56234133, 0.31622776, 0.17782794, 0.1, 0.05623413, 0.03162278, 0.01778279,
         0.01, 0.00562341, 0.00316228, 0.00177828, 0.001, 0.00056234, 0.00031623, 0.00017783],
        dtype=torch.float64,
    )
    assert torch.allclose(expected_inv_freq(16, 10000.0), published, rtol=1e-4, atol=1e-9)
    # A different theta must not also fit, or the assertion proves nothing.
    assert not torch.allclose(expected_inv_freq(16, 500000.0), published, rtol=1e-4, atol=1e-9)


def test_detect_source_layout_reads_a_real_shaped_tensor() -> None:
    heads, head_dim, cols = 8, 4, 6
    torch.manual_seed(1)
    parts = [torch.randn(heads * head_dim, cols) * s for s in (1.0, 0.6, 0.25)]
    contiguous = torch.cat(parts)
    interleaved = torch.cat(
        [p[h * head_dim : (h + 1) * head_dim] for h in range(heads) for p in parts]
    )
    assert detect_source_layout(contiguous, head_dim=head_dim) is RowLayout.CONTIGUOUS
    assert detect_source_layout(interleaved, head_dim=head_dim) is RowLayout.INTERLEAVED
