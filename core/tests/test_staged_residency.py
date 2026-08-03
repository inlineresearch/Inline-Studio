"""Splitting a modular pipeline so the conditioner does not hold VRAM through the denoise.

`pipeline_runtime` already does this for classic pipelines: encode, park the encoder on the CPU,
pass `prompt_embeds` back into the call. A modular pipeline has no `encode_prompt`, so the same
idea has to be expressed as a split by block name with a `PipelineState` handoff. These cover the
split itself, which is the part that can silently produce a pipeline missing a phase.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("diffusers.modular_pipelines")

from inline_core.models.pipeline_runtime import release_components, split_blocks  # noqa: E402


def _h3_blocks() -> Any:
    """The real H3 blockset. No weights are touched, only the block graph."""
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.vendor import MiniMaxH3Blocks

    return MiniMaxH3Blocks()


def test_the_conditioner_and_the_denoise_end_up_on_opposite_sides() -> None:
    """The whole point: the encoder can be freed between the two halves."""
    head, tail = split_blocks(_h3_blocks(), through="text_encoder")

    assert "text_encoder" in head.sub_blocks
    assert "denoise" in tail.sub_blocks and "decode" in tail.sub_blocks
    assert "text_encoder" not in tail.sub_blocks


def test_the_split_loses_no_phase_and_keeps_their_order() -> None:
    """A dropped block is the failure mode here, and it would surface as a render that is subtly
    wrong rather than one that raises."""
    blocks = _h3_blocks()
    original = list(blocks.sub_blocks)
    head, tail = split_blocks(blocks, through="text_encoder")

    assert list(head.sub_blocks) + list(tail.sub_blocks) == original


def test_a_name_that_is_not_a_block_is_refused() -> None:
    with pytest.raises(ValueError, match="not a block of this pipeline"):
        split_blocks(_h3_blocks(), through="encoder")


def test_splitting_at_the_last_block_is_refused() -> None:
    """Otherwise the second half is empty and the staged run silently becomes an ordinary one."""
    with pytest.raises(ValueError, match="nothing to run afterwards"):
        split_blocks(_h3_blocks(), through="decode")


class _Pipe:
    """A pipeline that keeps its own component map, which is what makes releasing subtle."""

    def __init__(self) -> None:
        self.text_encoder = object()
        self.transformer = object()
        self.registered = {"text_encoder": self.text_encoder}

    def update_components(self, **components: Any) -> None:
        for name, value in components.items():
            if value is None:
                self.registered.pop(name, None)
            else:
                self.registered[name] = value


def test_releasing_unregisters_rather_than_only_dereferencing() -> None:
    """A module still in the pipeline's map is still alive, however many names went out of scope,
    and that failure looks exactly like the release not happening."""
    pipe = _Pipe()

    release_components(pipe, ["text_encoder"])

    assert pipe.registered == {}
    assert pipe.text_encoder is None
    assert pipe.transformer is not None  # untouched


def test_releasing_a_component_that_is_not_there_is_not_an_error() -> None:
    pipe = _Pipe()
    release_components(pipe, ["audio_vae", "text_encoder", "audio_vae"])
    assert pipe.registered == {}
