"""Shrink MiniMax H3's AdaLN branch, which is 40 percent of the checkpoint.

Every block carries an ``adaln_proj.linear`` of ``[96768, 2688]``: 520 MB in bf16, 26 GB across the
50 blocks, against 40 GB for everything else. It exists to turn one timestep embedding into six
modulation vectors, and it is applied to ``silu(temb)`` and nothing else.

``silu(temb)`` turns out to live in a **five dimensional** subspace. Its singular values over the
whole timestep range are ``[23.0, 2.7, 1.7, 0.5, 0.4, 0, 0, ...]``, so a rank-8 basis captures it
exactly. Projecting through that basis first lets each block store ``[96768, 8]`` instead, which is
1.5 MB rather than 520 MB, and takes the transformer from 66.3 GB to about 40 GB.

MiniMax ship a pruned build that does the same thing with a 1025-row lookup table. This derives the
factorisation from the bf16 weights instead, which means:

* no dependency on their ``convrot`` format, which only ComfyUI reads,
* **no discrete timestep grid**. They index a table, so a sampler whose sigmas fall between rows
  needs snapping or interpolation. Projecting a continuous ``t`` through the basis is exact for any
  timestep, so the sampler is unconstrained.

Verified against their published tables: our factorisation reproduces the full-precision modulation
to 1.1e-4 relative error, theirs to 2e-4, and the two span the same subspace (principal angles 1.0
for the seven directions that carry any energy).

The vendored port is **not** edited for this. The factorised module is swapped in after the model is
built, so ``vendor/`` stays verbatim.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("inline_core.minimaxh3")

#: Rank kept. Five directions carry the energy; eight matches the published build and leaves slack.
RANK = 8

#: Points used to sample the timestep range when deriving the basis. The map is smooth and rank-5,
#: so this only has to be dense enough to span it; the result is then exact for continuous ``t``.
SAMPLES = 1025


@torch.no_grad()
def decompose(
    time_proj: nn.Module, time_embedder: nn.Module, *, samples: int = SAMPLES
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SVD of ``silu(temb)`` over the timestep range: singular values, and right vectors.

    Split out from ``derive_basis`` so the rank claim in this module's docstring is a number anyone
    can print from the real weights rather than a sentence to take on trust.
    """
    # A parameterless module is legitimate (a stub, an Identity), so placement falls back rather
    # than raising on an empty parameter list.
    reference = next(time_embedder.parameters(), None)
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = reference.dtype if reference is not None else torch.float32
    grid = torch.linspace(0, 1, samples, device=device)
    embedded = time_embedder(time_proj(grid).to(dtype))
    activated = torch.nn.functional.silu(embedded).float()
    _, singular, right = torch.linalg.svd(activated, full_matrices=False)
    return singular, right


@torch.no_grad()
def derive_basis(
    time_proj: nn.Module, time_embedder: nn.Module, *, rank: int = RANK, samples: int = SAMPLES
) -> torch.Tensor:
    """An orthonormal basis for ``silu(temb)``, as ``[time_embed_dim, rank]``.

    Sampled over ``t`` in [0, 1], which is the range H3's schedulers step across.
    """
    singular, right = decompose(time_proj, time_embedder, samples=samples)
    kept = float((singular[:rank] ** 2).sum() / (singular**2).sum())
    if kept < 0.999:
        raise ValueError(
            f"A rank-{rank} basis captures only {kept:.4f} of this timestep map, so "
            "factorising the AdaLN branch would change the model. Refusing rather than "
            "degrading it silently."
        )
    logger.info("AdaLN basis: rank %d captures %.6f of the timestep map", rank, kept)
    return right[:rank].T.contiguous()


class FactorisedAdaLN(nn.Module):
    """``adaln_proj`` with its projection taken through a low-rank basis.

    Same outputs as the module it replaces, in the same order, so nothing downstream changes.
    """

    def __init__(self, hidden_size: int, out_features: int, basis: torch.Tensor) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("basis", basis, persistent=True)
        self.linear = nn.Linear(basis.shape[1], out_features, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        activated = torch.nn.functional.silu(temb)
        projected = (activated.to(self.basis.dtype) @ self.basis).to(self.linear.weight.dtype)
        out = self.linear(projected).view(-1, 6 * self.hidden_size)
        return out.chunk(6, dim=-1)


@torch.no_grad()
def factorise_block(block: Any, basis: torch.Tensor) -> FactorisedAdaLN:
    """Replace one block's AdaLN projection with its factorised equivalent."""
    original = block.adaln_proj
    weight, bias = original.linear.weight, original.linear.bias
    # The basis is derived once for the whole model; under split residency the blocks it is applied
    # to are not all on the same device as the module it came from.
    basis = basis.to(device=weight.device)
    replacement = FactorisedAdaLN(original.hidden_size, weight.shape[0], basis.to(weight.dtype))
    # W @ V: the same map, expressed in the basis the input is projected into.
    replacement.linear.weight.data = (weight.float() @ basis.float()).to(weight.dtype)
    replacement.linear.bias.data = bias.detach().clone()
    return replacement.to(device=weight.device)


@torch.no_grad()
def factorise(model: Any, *, rank: int = RANK) -> int:
    """Factorise every transformer block's AdaLN branch. Returns the bytes saved."""
    basis = derive_basis(model.time_proj, model.time_embedder, rank=rank)
    saved = 0
    def nbytes(module: Any) -> int:
        weight = module.adaln_proj.linear.weight
        return weight.numel() * weight.element_size()

    for block in model.transformer_blocks:
        before = nbytes(block)
        block.adaln_proj = factorise_block(block, basis)
        saved += before - nbytes(block)
    logger.info(
        "AdaLN factorised: %d blocks, %.1f GB saved",
        len(model.transformer_blocks), saved / 1e9,
    )
    return saved
