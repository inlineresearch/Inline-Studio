"""Vendored MiniMax H3 model code, taken verbatim from an unmerged diffusers pull request.

    upstream: https://github.com/huggingface/diffusers
    pull:     #14355 "MiniMax-H3"
    branch:   minimax-h3
    commit:   abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
    taken:    2026-08-03

H3 is integrated upstream as Modular Diffusers blocks only, and that work is not in any diffusers
release, so there is nothing to depend on: the choice is to vendor it or to pin a moving branch.

**The only edit made to these files is import rewriting.** Relative imports that pointed into
diffusers were made absolute; the handful that reach for H3's own classes (which installed diffusers
does not define) now point at their siblings here. No reformatting, no renaming, no behaviour
changes, so re-syncing against the merged PR stays a diff a person can read.

``core/CLAUDE.md``'s file-length and comment-density rules do not apply inside this directory, and
``pyproject.toml`` excludes ``models/*/vendor/`` from ruff and pyright for the same reason: this is
third-party code, and editing it to satisfy our linters would destroy the property above.

Importing this package requires the ``runtime`` extra. That is deliberate and matches every other
model runner: an absent extra makes the import raise and ``server/bootstrap.py`` skips the model, so
a torch-less install still boots.
"""

from __future__ import annotations

from .autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from .autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from .modular_blocks_minimax_h3 import MiniMaxH3Blocks, MiniMaxH3Ref2VABlocks
from .modular_pipeline import MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline
from .packing_ref2va import MiniMaxH3Reference
from .scheduling_minimax_h3 import MiniMaxH3Scheduler
from .transformer_minimax_h3 import MiniMaxH3Transformer3DModel

__all__ = [
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3Blocks",
    "MiniMaxH3ModularPipeline",
    "MiniMaxH3Ref2VABlocks",
    "MiniMaxH3Ref2VAModularPipeline",
    "MiniMaxH3Reference",
    "MiniMaxH3Scheduler",
    "MiniMaxH3Transformer3DModel",
]
