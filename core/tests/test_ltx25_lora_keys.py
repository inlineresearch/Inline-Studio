"""Does a LoRA trained here load anywhere else, and at the same strength?

The published LTX convention is checked against the real published adapter where one is on disk, and
the scaling is checked against what each loader actually does with it. Both halves matter: a LoRA
under the wrong key does not load and says so, while a LoRA at the wrong scale loads silently and
just looks worse.
"""

from __future__ import annotations

import pytest

from inline_core.models.ltx25 import lora_keys
from inline_core.training import arch as archs

torch = pytest.importorskip("torch")


def peft_state(rank: int = 16, alpha: float = 32.0) -> dict[str, object]:
    """What `trainer._save_lora` hands to `export_keys`: PEFT factors plus its alpha scalars."""
    module = "transformer_blocks.0.attn1.to_q"
    return {
        f"{module}.lora_A.weight": torch.ones(rank, 8),
        f"{module}.lora_B.weight": torch.ones(8, rank),
        f"{module}.alpha": torch.tensor(alpha),
    }


def test_keys_take_the_published_prefix() -> None:
    out = lora_keys.export_reference(peft_state())
    assert all(k.startswith("diffusion_model.") for k in out)
    assert "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight" in out
    assert lora_keys.is_reference(out)


def test_the_alpha_scalars_do_not_survive() -> None:
    """`ltx_core` never reads them, and leaving them in would double-apply in our own loader once
    the scale is already folded into B."""
    out = lora_keys.export_reference(peft_state())
    assert not any(k.endswith(".alpha") for k in out)


def test_the_alpha_scale_is_folded_into_the_up_projection() -> None:
    """PEFT saves factors raw at a scale of alpha/rank. `ltx_core` applies only the user's strength,
    so an exported adapter has to already mean what it says."""
    rank, alpha = 16, 32.0
    out = lora_keys.export_reference(peft_state(rank, alpha))
    up = out["diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight"]
    assert torch.allclose(up, torch.full_like(up, alpha / rank))


def test_an_adapter_at_alpha_equals_rank_is_untouched() -> None:
    """The common case, and the one upstream ships: scale 1.0, so folding must be a no-op."""
    out = lora_keys.export_reference(peft_state(rank=16, alpha=16.0))
    up = out["diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight"]
    assert torch.allclose(up, torch.ones_like(up))


def test_the_delta_is_preserved_across_the_rescale() -> None:
    """The whole point: B@A times the scale must be what the model would have seen in training."""
    rank, alpha = 16, 32.0
    state = peft_state(rank, alpha)
    module = "transformer_blocks.0.attn1.to_q"
    trained = (state[f"{module}.lora_B.weight"] @ state[f"{module}.lora_A.weight"]) * (alpha / rank)

    out = lora_keys.export_reference(state)
    exported = (
        out[f"diffusion_model.{module}.lora_B.weight"]
        @ out[f"diffusion_model.{module}.lora_A.weight"]
    )
    assert torch.allclose(trained, exported), "an export must fuse to the trained delta at 1.0"


def test_the_arch_exports_through_this() -> None:
    assert archs.ARCHS[archs.LTX25].export_keys is not None
    out = archs.ARCHS[archs.LTX25].export_keys(peft_state())
    assert lora_keys.is_reference(out)


def test_our_own_loader_still_reads_what_we_export() -> None:
    """Portability is not one-way: the published prefix has to come back off on the way in."""
    from inline_core.models.lora import _PREFIXES

    assert lora_keys.PREFIX in _PREFIXES


# --- against the real published adapter, when it is on disk --------------------------------------


def published_lora():
    from inline_core.models.ltx25 import requirements as reqs

    path = reqs.resolve("loras", reqs.DISTILLED_LORA_FILE)
    if path is None:
        pytest.skip("the published distilled LoRA is not downloaded")
    header = reqs.read_header(path)
    assert header is not None
    return header


def test_the_published_adapter_uses_the_prefix_we_write() -> None:
    header = published_lora()
    keys = [k for k in header if k != "__metadata__"]
    assert keys and all(k.startswith(lora_keys.PREFIX) for k in keys)


def test_the_published_adapter_uses_lora_a_and_b_not_lora_down_up() -> None:
    """Two conventions exist in the wild; LTX publishes the diffusers one, which is also PEFT's."""
    keys = [k for k in published_lora() if k != "__metadata__"]
    assert any(k.endswith(".lora_A.weight") for k in keys)
    assert not any(".lora_down." in k for k in keys)


def test_the_published_adapter_carries_no_alpha_tensors() -> None:
    """It ships rank == alpha in metadata instead, which is why upstream never reads one."""
    keys = [k for k in published_lora() if k != "__metadata__"]
    assert not any(k.endswith(".alpha") for k in keys)
