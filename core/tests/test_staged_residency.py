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


def test_both_h3_blocksets_split_at_the_same_point() -> None:
    """FL2VA and Ref2VA differ after the encoder - `vae_encoder` against `reference_encoder` - but
    staging cuts at `text_encoder`, so both must carry that block under that name. A rename would
    break staging for one partition only, which is the kind of thing that ships."""
    from inline_core.models.minimaxh3.vendor import MiniMaxH3Blocks, MiniMaxH3Ref2VABlocks

    for blocks in (MiniMaxH3Blocks(), MiniMaxH3Ref2VABlocks()):
        head, tail = split_blocks(blocks, through="text_encoder")
        assert list(head.sub_blocks) == ["setup", "text_encoder"]
        assert "denoise" in tail.sub_blocks


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


# --- cancellation on the staged path ------------------------------------------------------------


class _Module:
    """Records every device it is moved to, so the transfer traffic is visible to a test."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: Any) -> _Module:
        self.moves.append(str(device))
        return self


class _Phase:
    def __init__(self, result: Any = "state", boom: Any = None) -> None:
        self.result = result
        self.boom = boom
        self.calls = 0

    def __call__(self, *_args: Any, **_kw: Any) -> Any:
        self.calls += 1
        if self.boom is not None:
            raise self.boom
        return self.result


class _StagedPipe:
    def __init__(self, head: _Phase, tail: _Phase) -> None:
        self._inline_phases = (head, tail)
        self._inline_phase_owners = ("encoder", "denoiser")
        self.text_encoder = _Module()
        self.transformer = _Module()


def _render(pipe: Any, **kw: Any) -> Any:
    from inline_core.models.minimaxh3.pipeline import render_staged

    return render_staged(pipe, "cuda:0", **kw)


def test_a_cancelled_denoise_does_not_pay_the_restore_transfers() -> None:
    """The old `finally` moved the denoiser off and the conditioner back before the exception got
    out - about 40GB of PCIe traffic, which read as the cancel being ignored."""
    from inline_core.errors import CancelledError

    pipe = _StagedPipe(_Phase(), _Phase(boom=CancelledError("Run cancelled.")))
    with pytest.raises(CancelledError):
        _render(pipe)

    # Setup moves only: the denoiser parked, the conditioner up then parked, the denoiser placed.
    assert pipe.transformer.moves == ["cpu", "cuda:0"]
    assert pipe.text_encoder.moves == ["cuda:0", "cpu"]


def test_each_half_claims_the_card_before_it_runs() -> None:
    pipe = _StagedPipe(_Phase(), _Phase(result="video"))
    assert _render(pipe) == "video"
    # The conditioner never holds the card while the denoiser does, in either direction.
    assert pipe.transformer.moves == ["cpu", "cuda:0"]
    assert pipe.text_encoder.moves == ["cuda:0", "cpu"]


def test_cancelling_before_the_encode_skips_both_halves() -> None:
    from inline_core.errors import CancelledError

    head, tail = _Phase(), _Phase()

    def cancel_check() -> None:
        raise CancelledError("Run cancelled.")

    with pytest.raises(CancelledError):
        _render(_StagedPipe(head, tail), cancel_check=cancel_check)
    assert head.calls == 0 and tail.calls == 0


def test_cancelling_after_the_encode_skips_the_denoise() -> None:
    """The 32B conditioner runs with no step hook, so this is the only checkpoint covering it."""
    from inline_core.errors import CancelledError

    head, tail = _Phase(), _Phase()
    seen = {"n": 0}

    def cancel_check() -> None:
        seen["n"] += 1
        if seen["n"] > 1:  # let the pre-encode check pass, fail the post-encode one
            raise CancelledError("Run cancelled.")

    with pytest.raises(CancelledError):
        _render(_StagedPipe(head, tail), cancel_check=cancel_check)
    assert head.calls == 1 and tail.calls == 0


# --- the phase split ------------------------------------------------------------------------------


def test_no_staged_phase_holds_a_component_it_never_reads() -> None:
    """Held as one tail, the vision tower encode carried a 10.4 GB VAE it never touches, which on a
    45 GB card was the difference between rendering and an OOM. Derived from `expected_components`
    so it keeps holding if the blockset is reordered."""
    pytest.importorskip("diffusers")
    from inline_core.models.minimaxh3.pipeline import _denoiser_name, _staged_phases
    from inline_core.models.minimaxh3.vendor import MiniMaxH3Blocks, MiniMaxH3Ref2VABlocks

    for blocks in (MiniMaxH3Ref2VABlocks(), MiniMaxH3Blocks()):
        # `transformer_ref` on ref2va, `transformer` on fl2va.
        big = {"encoder": "text_encoder", "vae": "vae", "denoiser": _denoiser_name(blocks)}
        owners, parts = _staged_phases(blocks)
        assert len(parts) == 4
        for owner, part in zip(owners, parts, strict=True):
            declared: set[str] = set()
            for block in part.sub_blocks.values():
                declared |= {
                    getattr(spec, "name", spec) for spec in (block.expected_components or [])
                }
            for role, component in big.items():
                held = component in declared
                if role == owner:
                    assert held, f"the {owner} phase does not read {component}"
                else:
                    assert not held, f"the {owner} phase would hold {component} unused"


class _StagedPipeVAE:
    """A four phase pipe whose three large components each record where they are moved."""

    def __init__(self, phases: tuple[_Phase, ...]) -> None:
        self._inline_phases = phases
        self._inline_phase_owners = ("encoder", "vae", "denoiser", "vae")
        self._inline_staged_vae = True
        self.text_encoder = _Module()
        self.transformer = _Module()
        self.vae = _Module()


def test_the_vae_is_off_the_card_for_the_encode_and_the_denoise() -> None:
    """Exactly the two phases that do not read it, and the two that do get it back."""
    pipe = _StagedPipeVAE(tuple(_Phase(result=f"s{i}") for i in range(4)))
    assert _render(pipe) == "s3"

    # Parked for phase 1, placed for 2, parked for 3, placed again for 4.
    assert pipe.vae.moves == ["cpu", "cuda:0", "cpu", "cuda:0"]
    # The conditioner goes up once and never comes back; the denoiser only for its own phase.
    assert pipe.text_encoder.moves == ["cuda:0", "cpu"]
    assert pipe.transformer.moves == ["cpu", "cuda:0", "cpu"]


def test_a_streamed_vae_is_never_moved_by_the_phase_loop() -> None:
    """Without `_inline_staged_vae` it carries offload hooks, and a `.to()` fights them."""
    pipe = _StagedPipeVAE(tuple(_Phase() for _ in range(4)))
    pipe._inline_staged_vae = False
    _render(pipe)
    assert pipe.vae.moves == []
