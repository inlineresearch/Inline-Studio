"""MiniMax H3 LoRA key translation, both directions.

Every transform here fails silently when it is wrong - a mis-split QKV or a backwards gated FFN
still loads and still renders - so these tests check the *delta each module receives*, not the key
names. A rename test would pass on all of them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from inline_core.errors import ComponentError  # noqa: E402
from inline_core.models.minimaxh3.lora_keys import (  # noqa: E402
    adapt,
    export_reference,
    import_reference,
    is_reference,
)

RANK = 4
HEADS = 2
HEAD_DIM = 128
PART = HEADS * HEAD_DIM  # rows one of q, k or v contributes
FFN = 16  # rows of the gated FFN's fused pair, half gate and half value
WIDTH = 8


def _pair(rows: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "down": torch.randn(RANK, WIDTH, generator=g),
        "up": torch.randn(rows, RANK, generator=g),
    }


def _ours() -> dict[str, torch.Tensor]:
    """A one-block adapter over every module H3 training targets."""
    modules = {
        "transformer_blocks.0.attn.to_q": PART,
        "transformer_blocks.0.attn.to_k": PART,
        "transformer_blocks.0.attn.to_v": PART,
        "transformer_blocks.0.attn.to_out.0": PART,
        "transformer_blocks.0.ff.net.0.proj": FFN,
        "transformer_blocks.0.ff.net.2": FFN,
        "token_refiner.refiner_blocks.1.attn.to_q": PART,
        "token_refiner.refiner_blocks.1.attn.to_k": PART,
        "token_refiner.refiner_blocks.1.attn.to_v": PART,
        "context_embedder": WIDTH,
    }
    state: dict[str, torch.Tensor] = {}
    for seed, (name, rows) in enumerate(modules.items()):
        for role, suffix in (("down", "lora_A"), ("up", "lora_B")):
            state[f"base_model.model.{name}.{suffix}.weight"] = _pair(rows, seed)[role]
    return state


def _deltas(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``B @ A`` per module stem, which is the only thing the model ever sees."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith("lora_B.weight"):
            stem = key[: -len(".lora_B.weight")]
            out[stem] = value @ state[f"{stem}.lora_A.weight"]
    return out


def test_export_uses_the_reference_key_names() -> None:
    keys = set(export_reference(_ours()))
    assert "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight" in keys
    assert "diffusion_model.blocks.0.mlp.fc1.lora_B.weight" in keys
    assert "diffusion_model.blocks.0.mlp.fc2.lora_B.weight" in keys
    assert "diffusion_model.blocks.0.attn.out_proj.lora_B.weight" in keys
    assert "diffusion_model.token_refiner.blocks.1.attn.qkv_proj.lora_B.weight" in keys
    assert "diffusion_model.condition_proj.lora_B.weight" in keys
    # The split names must be gone, or a tool that matches loosely applies the LoRA twice.
    assert not [k for k in keys if "to_q" in k or "transformer_blocks" in k]


def test_fused_qkv_carries_the_same_delta_as_the_three_it_replaces() -> None:
    ours = _ours()
    exported = _deltas(export_reference(ours))
    mine = _deltas(ours)
    stacked = torch.cat(
        [mine[f"base_model.model.transformer_blocks.0.attn.to_{p}"] for p in "qkv"], dim=0
    )
    fused = exported["diffusion_model.blocks.0.attn.qkv_proj"]
    assert fused.shape == (3 * PART, WIDTH)
    assert torch.allclose(fused, stacked, atol=1e-6)


def test_round_trip_returns_every_original_delta() -> None:
    ours = _ours()
    back = _deltas(import_reference(export_reference(ours)))
    for stem, delta in _deltas(ours).items():
        name = stem.removeprefix("base_model.model.")
        assert torch.allclose(back[name], delta, atol=1e-6), name
    assert set(back) == {s.removeprefix("base_model.model.") for s in _deltas(ours)}


def test_round_trip_survives_the_interleaved_layout() -> None:
    ours = _ours()
    exported = export_reference(ours, target="minimaxai")
    back = _deltas(import_reference(exported, source="minimaxai"))
    for stem, delta in _deltas(ours).items():
        assert torch.allclose(back[stem.removeprefix("base_model.model.")], delta, atol=1e-6)


def test_the_two_layouts_are_not_the_same_bytes() -> None:
    """If they were, the layout argument would be decoration and a wrong guess would be harmless."""
    key = "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"
    ours = _ours()
    assert not torch.equal(
        export_reference(ours, target="comfy-org")[key],
        export_reference(ours, target="minimaxai")[key],
    )


def test_reading_a_layout_the_wrong_way_round_corrupts_the_delta() -> None:
    """The failure this defaults around: no error, just the wrong weights."""
    ours = _ours()
    wrong = _deltas(import_reference(export_reference(ours, target="minimaxai")))
    right = _deltas(ours)["base_model.model.transformer_blocks.0.attn.to_q"]
    assert not torch.allclose(wrong["transformer_blocks.0.attn.to_q"], right, atol=1e-6)


def test_gated_ffn_halves_are_exchanged_and_exchanged_back() -> None:
    ours = _ours()
    mine = _deltas(ours)["base_model.model.transformer_blocks.0.ff.net.0.proj"]
    theirs = _deltas(export_reference(ours))["diffusion_model.blocks.0.mlp.fc1"]
    half = FFN // 2
    assert torch.allclose(theirs[:half], mine[half:], atol=1e-6)
    assert torch.allclose(theirs[half:], mine[:half], atol=1e-6)


def test_alpha_triples_with_the_rank_so_the_scale_is_unchanged() -> None:
    ours = _ours()
    for part in "qkv":
        ours[f"base_model.model.transformer_blocks.0.attn.to_{part}.alpha"] = torch.tensor(2.0)
    exported = export_reference(ours)
    alpha = exported["diffusion_model.blocks.0.attn.qkv_proj.alpha"]
    rank = exported["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].shape[0]
    assert rank == 3 * RANK
    assert float(alpha) / rank == pytest.approx(2.0 / RANK)


def test_the_effective_scale_survives_the_split_back() -> None:
    """``alpha / rank`` is what the fuser multiplies by, so that ratio is the invariant."""
    ours = _ours()
    for part in "qkv":
        ours[f"base_model.model.transformer_blocks.0.attn.to_{part}.alpha"] = torch.tensor(2.0)
    back = import_reference(export_reference(ours))
    alpha = float(back["transformer_blocks.0.attn.to_q.alpha"])
    rank = back["transformer_blocks.0.attn.to_q.lora_A.weight"].shape[0]
    assert alpha / rank == pytest.approx(2.0 / RANK)


def test_a_third_party_rank_r_adapter_splits_to_rank_r() -> None:
    """The common import: a fused rank-r adapter must not be inflated on the way in."""
    theirs = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight": torch.randn(RANK, WIDTH),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight": torch.randn(3 * PART, RANK),
        "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
    }
    out = import_reference(theirs)
    assert out["transformer_blocks.0.attn.to_q.lora_A.weight"].shape[0] == RANK
    assert float(out["transformer_blocks.0.attn.to_q.alpha"]) == pytest.approx(2.0)


def test_comfy_lora_down_up_naming_is_read() -> None:
    theirs = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight": torch.randn(RANK, WIDTH),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight": torch.randn(3 * PART, RANK),
    }
    out = import_reference(theirs)
    assert set(out) == {
        f"transformer_blocks.0.attn.to_{p}.lora_{s}.weight" for p in "qkv" for s in "AB"
    }
    assert torch.equal(
        out["transformer_blocks.0.attn.to_k.lora_B.weight"],
        theirs["diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight"][PART : 2 * PART],
    )


def test_the_three_split_parts_share_one_lora_a() -> None:
    theirs = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight": torch.randn(RANK, WIDTH),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight": torch.randn(3 * PART, RANK),
    }
    out = import_reference(theirs)
    first = out["transformer_blocks.0.attn.to_q.lora_A.weight"]
    for part in "kv":
        assert torch.equal(out[f"transformer_blocks.0.attn.to_{part}.lora_A.weight"], first)


def test_exporting_a_partial_attention_is_refused() -> None:
    ours = _ours()
    for suffix in ("lora_A", "lora_B"):
        del ours[f"base_model.model.transformer_blocks.0.attn.to_v.{suffix}.weight"]
    with pytest.raises(ComponentError, match="2 of 3"):
        export_reference(ours)


def test_an_adapter_for_another_model_is_refused() -> None:
    with pytest.raises(ComponentError, match="different model"):
        import_reference(
            {"diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros(4, 8)}
        )


def test_an_unknown_layout_names_the_ones_that_exist() -> None:
    with pytest.raises(ComponentError, match="comfy-org"):
        import_reference(_ours(), source="nonsense")


def test_adapt_passes_our_own_adapter_through_untouched() -> None:
    """Old files, and every fresh non-H3 one, must not be rewritten on the way in."""
    ours = _ours()
    assert adapt(ours) is ours
    assert not is_reference(ours)


def test_adapt_translates_a_reference_keyed_adapter() -> None:
    theirs = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight": torch.randn(RANK, WIDTH),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight": torch.randn(3 * PART, RANK),
    }
    assert is_reference(theirs)
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in adapt(theirs)


def test_a_feed_forward_only_adapter_is_still_recognised() -> None:
    """Detection is on the block prefix: an adapter that never touched attention has no fused
    tensor to give it away."""
    assert is_reference(
        {
            "diffusion_model.blocks.3.mlp.fc2.lora_down.weight": torch.randn(RANK, WIDTH),
            "diffusion_model.blocks.3.mlp.fc2.lora_up.weight": torch.randn(FFN, RANK),
        }
    )


def test_a_round_trip_through_save_and_load_keeps_the_delta() -> None:
    """What actually ships: train here, write the portable file, read it back here."""
    ours = _ours()
    back = _deltas(adapt(export_reference(ours)))
    for stem, delta in _deltas(ours).items():
        assert torch.allclose(back[stem.removeprefix("base_model.model.")], delta, atol=1e-6)
