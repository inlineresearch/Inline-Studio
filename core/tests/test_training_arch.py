"""The per-architecture training behaviour.

The flow-matching conventions are the sharp edge: Z-Image and Krea 2 use **opposite signs and
opposite timestep directions**, and getting one backwards trains a LoRA that quietly makes output
worse rather than erroring. These are cheap to pin, so they are pinned.
"""

from __future__ import annotations

import pytest

from inline_core.training import arch as archs

torch = pytest.importorskip("torch")


def test_defaults_to_z_image_so_older_runs_resume() -> None:
    assert archs.get(None).key == archs.Z_IMAGE
    assert archs.get("").key == archs.Z_IMAGE


def test_unknown_arch_is_refused() -> None:
    with pytest.raises(RuntimeError, match="Unknown training architecture"):
        archs.get("stable-diffusion-1.5")


def test_the_two_archs_have_opposite_flow_conventions() -> None:
    clean, noise = torch.ones(2), torch.zeros(2)
    zimage, krea2 = archs.get(archs.Z_IMAGE), archs.get(archs.KREA2)
    sigma = torch.tensor(0.3)

    assert torch.equal(zimage.target(clean, noise), clean - noise)
    assert torch.equal(krea2.target(clean, noise), noise - clean)
    # Z-Image counts 1 = clean; Krea 2 hands the raw noise fraction straight through.
    assert zimage.timestep(sigma) == pytest.approx(0.7)
    assert krea2.timestep(sigma) == pytest.approx(0.3)


def test_sigma_is_a_scalar_noise_fraction_in_the_unit_interval() -> None:
    for key in (archs.Z_IMAGE, archs.KREA2):
        for _ in range(50):
            sigma = archs.get(key).sigma("cpu", 3.0)
            assert sigma.shape == ()
            assert 0.0 < float(sigma) < 1.0


def test_krea2_targets_match_the_authors_recommended_set() -> None:
    targets = archs.get(archs.KREA2).target_modules

    # From diffusers' train_dreambooth_lora_krea2.py; the text-fusion stage and the img/txt/time
    # projections matter as much as attention for Krea 2.
    assert "text_fusion.projector" in targets
    assert "time_mod_proj" in targets
    assert "img_in" in targets
    assert {"to_q", "to_k", "to_v", "to_out.0", "to_gate"} <= set(targets)
    # Z-Image's SwiGLU names must not leak in - they match nothing in Krea 2.
    assert not {"w1", "w2", "w3"} & set(targets)


def test_zimage_targets_are_unchanged() -> None:
    assert archs.get(archs.Z_IMAGE).target_modules == [
        "to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3",
    ]


class _EchoTransformer:
    """Returns the packed hidden states unchanged, so pack/unpack must round trip to the input."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def __call__(self, **kwargs: object) -> tuple[object]:
        self.seen = kwargs
        return (kwargs["hidden_states"],)


def test_krea2_forward_packs_and_unpacks_back_to_the_latent_grid() -> None:
    pytest.importorskip("diffusers")
    latent = torch.arange(16 * 8 * 8, dtype=torch.float32).reshape(16, 8, 8)
    item = {"embed": torch.zeros(7, 12, 4), "mask": torch.ones(7, dtype=torch.bool)}
    transformer = _EchoTransformer()

    out = archs.get(archs.KREA2).forward(transformer, latent, torch.tensor(0.5), item)

    assert torch.equal(out, latent)
    # 2x2 patches: a 16x8x8 latent becomes 16 tokens of 64 channels, after 7 text tokens.
    assert tuple(transformer.seen["hidden_states"].shape) == (1, 16, 64)
    assert tuple(transformer.seen["position_ids"].shape) == (7 + 16, 3)
    assert transformer.seen["encoder_attention_mask"].dtype is torch.bool


def test_attention_scope_narrows_to_the_projections_each_arch_has() -> None:
    krea2 = archs.get(archs.KREA2)
    zimage = archs.get(archs.Z_IMAGE)

    assert archs.target_modules(krea2, "attention") == [
        "to_q", "to_k", "to_v", "to_out.0", "to_gate",
    ]
    # Z-Image has no gate projection, so the same scope resolves to what it actually has.
    assert archs.target_modules(zimage, "attention") == ["to_q", "to_k", "to_v", "to_out.0"]


def test_full_scope_is_the_default_and_unchanged() -> None:
    krea2 = archs.get(archs.KREA2)

    assert archs.target_modules(krea2, "full") == krea2.target_modules
    assert archs.target_modules(krea2, None) == krea2.target_modules
    assert archs.target_modules(krea2, "") == krea2.target_modules


def test_an_unknown_scope_is_refused_rather_than_silently_training_everything() -> None:
    with pytest.raises(RuntimeError, match="Unknown LoRA scope"):
        archs.target_modules(archs.get(archs.KREA2), "attn")
