"""LTX-2.5's training arch: the flow convention, the clip grid, and which Linears each mode adapts.

The convention is the dangerous part. Getting it wrong does not raise - it trains an adapter that
quietly degrades output - so it is derived here from the vendored `to_velocity` / `to_denoised` pair
rather than restated from the arch table it is meant to check.
"""

from __future__ import annotations

import pytest

from inline_core.training import arch as archs

torch = pytest.importorskip("torch")

ARCH = archs.ARCHS[archs.LTX25]


def test_the_target_is_the_velocity_upstream_would_compute() -> None:
    """`noisy` is built exactly as `trainer.py` builds it, and the target must be the velocity
    upstream's own helper derives from that same sample."""
    from inline_core.models.ltx25.vendor.ltx_core.utils import to_velocity

    torch.manual_seed(0)
    clean, noise = torch.randn(4, 8), torch.randn(4, 8)
    sigma = torch.tensor(0.37)
    noisy = (1 - sigma) * clean + sigma * noise

    assert torch.allclose(
        ARCH.target(clean, noise), to_velocity(noisy, sigma, clean), atol=1e-5
    )


def test_the_prediction_round_trips_back_to_the_clean_latent() -> None:
    """The other half of the same claim: feeding the target back through upstream's denoiser has to
    recover what the model was asked to reconstruct."""
    from inline_core.models.ltx25.vendor.ltx_core.utils import to_denoised

    torch.manual_seed(1)
    clean, noise = torch.randn(4, 8), torch.randn(4, 8)
    sigma = torch.tensor(0.62)
    noisy = (1 - sigma) * clean + sigma * noise

    recovered = to_denoised(noisy, ARCH.target(clean, noise), sigma)
    assert torch.allclose(recovered, clean, atol=1e-5)


def test_the_timestep_is_the_sigma_itself() -> None:
    """Krea 2's and FLUX.2's convention, not Z-Image's inverted one."""
    sigma = torch.tensor(0.25)
    assert torch.allclose(ARCH.timestep(sigma), sigma)
    assert torch.allclose(
        archs.ARCHS[archs.KREA2].timestep(sigma), ARCH.timestep(sigma)
    )
    assert not torch.allclose(
        archs.ARCHS[archs.Z_IMAGE].timestep(sigma), ARCH.timestep(sigma)
    )


# --- the clip grid -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "frames"),
    [(1.0, 17), (2.0, 41), (5.0, 113), (0.1, 9)],
)
def test_clip_length_snaps_down_onto_the_eight_plus_one_grid(
    seconds: float, frames: int
) -> None:
    assert archs.clip_frames(ARCH, seconds) == frames
    assert (frames - 1) % 8 == 0


def test_the_clip_floor_is_one_chunk_plus_the_head() -> None:
    assert ARCH.clip is not None
    assert ARCH.clip.min_frames == 9


def test_training_never_asks_for_frames_a_clip_does_not_hold() -> None:
    """Generation snaps up; training must not, or the VAE is handed frames the file never had."""
    assert ARCH.clip is not None
    for frames in range(9, 500):
        assert ARCH.clip.snap(frames) <= frames


def test_the_generation_and_training_grids_describe_the_same_vae() -> None:
    """Two types, opposite rounding, one model. They may differ in direction but never in shape."""
    from inline_core.models.ltx25.runner import GRID

    assert ARCH.clip is not None
    assert (ARCH.clip.fps, ARCH.clip.grid, ARCH.clip.offset) == (
        GRID.fps, GRID.grid, GRID.offset
    )


# --- target modules ------------------------------------------------------------------------------


def test_a_clip_lora_reaches_the_audio_and_cross_modal_branches() -> None:
    """Short patterns match by suffix, so `to_k` reaches `attn1`, `attn2`, `audio_attn*` and the two
    cross-modal blocks. That reach is what keeps the soundtrack in step with the adapter."""
    modules = archs.target_modules(ARCH, "full", archs.MODE_CLIP)
    assert modules == ["to_k", "to_q", "to_v", "to_out.0"]
    assert not any(m.startswith("attn1.") for m in modules)


def test_a_motion_lora_leaves_the_audio_branch_alone() -> None:
    """An IC-LoRA learns a video-to-video transform, so the audio branch is not its business."""
    modules = archs.target_modules(ARCH, "full", archs.MODE_CONTROL)
    assert all(m.startswith(("attn1.", "attn2.", "ff.")) for m in modules)
    assert "ff.net.0.proj" in modules


def test_narrowing_to_attention_works_on_both_modes() -> None:
    """`attention` has to mean the same thing whichever mode is selected, including when the mode's
    patterns carry an `attn1.` prefix the shared table does not."""
    for mode in (archs.MODE_CLIP, archs.MODE_CONTROL):
        narrowed = archs.target_modules(ARCH, "attention", mode)
        assert narrowed
        assert not any("ff." in m for m in narrowed)


def test_an_unknown_mode_falls_back_to_the_arch_default() -> None:
    assert archs.target_modules(ARCH, "full", None) == ARCH.target_modules


# --- latent tools --------------------------------------------------------------------------------


def test_latent_tools_are_derived_from_the_shape_in_hand() -> None:
    """`VideoLatentTools` asserts the latent matches its `target_shape`, so carrying one on the item
    would turn a mixed-shape dataset into an assertion. It is built from the latent instead."""
    from inline_core.training.arch import ltx25_latent_tools

    tools = ltx25_latent_tools((1, 128, 7, 18, 30))
    state = tools.create_initial_state("cpu", torch.float32, torch.zeros(1, 128, 7, 18, 30))
    assert state is not None
    assert tools.patchifier.get_token_count(tools.target_shape) > 0


def test_a_second_shape_gets_its_own_tools() -> None:
    from inline_core.training.arch import ltx25_latent_tools

    a = ltx25_latent_tools((1, 128, 7, 18, 30))
    b = ltx25_latent_tools((1, 128, 13, 18, 30))
    assert a is not b
    assert a is ltx25_latent_tools((1, 128, 7, 18, 30)), "same shape reuses the cached tools"


def test_the_wrong_shape_is_refused_rather_than_reshaped() -> None:
    """The assertion is the safety net: a latent that does not match would otherwise be patchified
    against the wrong grid and train on nonsense."""
    from inline_core.training.arch import ltx25_latent_tools

    tools = ltx25_latent_tools((1, 128, 7, 18, 30))
    with pytest.raises(AssertionError):
        tools.create_initial_state("cpu", torch.float32, torch.zeros(1, 128, 9, 18, 30))
