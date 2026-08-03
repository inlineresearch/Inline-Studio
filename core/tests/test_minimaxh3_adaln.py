"""Factorising the AdaLN branch, which is 40 percent of MiniMax H3's checkpoint.

The claim is that ``silu(temb)`` is low-rank, so each block's ``[96768, 2688]`` projection can be
replaced by ``[96768, 8]`` without changing what the model computes. That is a numerical claim, so
it is tested numerically: on a synthetic timestep map here, and against the real weights and
MiniMax's own published tables in the network-gated test at the bottom.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from inline_core.models.minimaxh3.adaln import (  # noqa: E402
    RANK,
    FactorisedAdaLN,
    derive_basis,
    factorise_block,
)


class _Proj(nn.Module):
    """Stands in for `Timesteps`: a smooth, genuinely low-rank map from t to features.

    Few frequencies on purpose. A scalar t traces a one-dimensional curve, and it is the smoothness
    of that curve that makes a handful of principal components capture it. H3's real map is rank-5;
    a wigglier stand-in would leave residual energy and test a weaker property than the real one.
    """

    def __init__(self, dim: int = 6) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, self.dim // 2 + 1, dtype=torch.float32, device=t.device)
        angles = t[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _Block(nn.Module):
    """The shape of a real block's AdaLN branch: one projection, six chunked outputs."""

    def __init__(self, hidden: int, embed: int) -> None:
        super().__init__()
        self.adaln_proj = nn.Module()
        self.adaln_proj.hidden_size = hidden
        self.adaln_proj.linear = nn.Linear(embed, 6 * hidden * 3)


def _pieces(hidden: int = 4, embed: int = 32):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    proj = _Proj()
    embedder = nn.Sequential(nn.Linear(6, embed), nn.SiLU(), nn.Linear(embed, embed))
    return proj, embedder, _Block(hidden, embed)


def _original(block, temb):  # type: ignore[no-untyped-def]
    out = block.adaln_proj.linear(torch.nn.functional.silu(temb))
    return out.view(-1, 6 * block.adaln_proj.hidden_size).chunk(6, dim=-1)


def test_the_basis_spans_the_timestep_map() -> None:
    proj, embedder, _ = _pieces()
    basis = derive_basis(proj, embedder)
    assert basis.shape[1] == RANK
    assert basis.shape[0] == 32
    # Orthonormal columns, which is what makes the projection a change of basis rather than a fit.
    assert torch.allclose(basis.T @ basis, torch.eye(RANK), atol=1e-4)


def test_factorising_preserves_the_output() -> None:
    """The whole point: same numbers out, far fewer weights in."""
    proj, embedder, block = _pieces()
    basis = derive_basis(proj, embedder)
    temb = embedder(proj(torch.linspace(0, 1, 17)))

    before = _original(block, temb)
    block.adaln_proj = factorise_block(block, basis)
    after = block.adaln_proj(temb)

    assert len(after) == len(before) == 6
    for got, want in zip(after, before, strict=True):
        assert torch.allclose(got, want, atol=1e-3, rtol=1e-3)


def test_it_holds_for_timesteps_the_basis_never_saw() -> None:
    """A lookup table would need snapping or interpolation here; a projection does not."""
    proj, embedder, block = _pieces()
    basis = derive_basis(proj, embedder, samples=64)
    odd = torch.tensor([0.01234, 0.37771, 0.5, 0.98765])
    temb = embedder(proj(odd))

    before = _original(block, temb)
    block.adaln_proj = factorise_block(block, basis)

    for got, want in zip(block.adaln_proj(temb), before, strict=True):
        assert torch.allclose(got, want, atol=1e-3, rtol=1e-3)


def test_the_replacement_is_far_smaller() -> None:
    proj, embedder, block = _pieces(hidden=4, embed=32)
    before = block.adaln_proj.linear.weight.numel()
    block.adaln_proj = factorise_block(block, derive_basis(proj, embedder))
    assert block.adaln_proj.linear.weight.numel() == before // 32 * RANK
    assert isinstance(block.adaln_proj, FactorisedAdaLN)


def test_a_map_that_is_not_low_rank_is_refused() -> None:
    """Silently degrading the model would be worse than not shrinking it."""

    class _Full(nn.Module):
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            torch.manual_seed(1)
            return torch.randn(t.shape[0], 64)

    with pytest.raises(ValueError, match="Refusing rather than degrading"):
        derive_basis(_Full(), nn.Identity(), rank=2, samples=128)


@pytest.mark.skipif(
    not os.environ.get("INLINE_H3_WEIGHTS"),
    reason="set INLINE_H3_WEIGHTS=1 with the real checkpoint present",
)
def test_against_the_real_weights_and_the_published_tables() -> None:
    """The claim, on the actual model: our factorisation reproduces the full-precision modulation
    at least as well as MiniMax's own pruned build does."""
    from pathlib import Path

    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from safetensors import safe_open

    source = Path("models/diffusion_models/minimax_h3_fl2va_bf16.safetensors")
    if not source.is_file():
        pytest.skip("bf16 checkpoint not present")
    with safe_open(str(source), framework="pt") as handle:
        get = lambda k: handle.get_tensor(k).float()  # noqa: E731
        embedder = TimestepEmbedding(in_channels=256, time_embed_dim=5376, out_dim=2688)
        embedder.linear_1.weight.data = get("time_embedder.proj_in.weight")
        embedder.linear_1.bias.data = get("time_embedder.proj_in.bias")
        embedder.linear_2.weight.data = get("time_embedder.proj_out.weight")
        embedder.linear_2.bias.data = get("time_embedder.proj_out.bias")
        weight = get("blocks.0.adaln_proj.linear.weight")

    proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    basis = derive_basis(proj, embedder)
    activated = torch.nn.functional.silu(embedder(proj(torch.linspace(0, 1, 1025))))

    exact = activated @ weight.T
    ours = (activated @ basis) @ (weight @ basis).T
    error = float((exact - ours).norm() / exact.norm())

    assert error < 1e-3, f"rank-{RANK} factorisation lost too much: {error}"
    # 96768x2688 down to 96768x8: the reason any of this is worth doing.
    assert weight.shape[1] // RANK == 336
