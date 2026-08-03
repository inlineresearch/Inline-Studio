"""The offload recipe map and the prepared-weight cache. Both run with no GPU and no torchao."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inline_core.device.policy import FitEstimate, OffloadMode, Profile, Quantization
from inline_core.models.offload import (
    BLOCK_LEVEL,
    LEAF_LEVEL,
    blocks_that_fit,
    blocks_to_place,
    describe,
    recipe_for,
)
from inline_core.models.prepared import PreparedKey, lookup, prepared_dir, prune, publish

#: Read off the published int8 build rather than guessed: only these four families were quantised.
H3_KEEP = ("proj_in", "audio_proj_in", "context_embedder", "time_embedder", "token_refiner")


class _Policy:
    """Just enough policy to answer `fit_estimate`; nothing else is consulted."""

    def __init__(self, plan: str | None) -> None:
        self._plan = plan

    def fit_estimate(self) -> Any:
        if self._plan is None:
            return None
        return FitEstimate(
            plan=self._plan,
            quant=Quantization.NONE,
            offload_mode=OffloadMode.NONE,
            profile=Profile.CPU,
            required_vram_gb=1.0,
            total_vram_gb=24.0,
            fits=self._plan != "wont-fit",
            note="",
        )


# --- the recipe map -------------------------------------------------------------------------------


def test_resident_neither_quantises_nor_offloads() -> None:
    recipe = recipe_for(_Policy("resident"), keep_precision=H3_KEEP)
    assert not recipe.quantizes
    assert recipe.denoiser_offload is None and recipe.vae_offload is None


def test_int8_keeps_the_denoiser_resident_and_keeps_the_exclusion_list() -> None:
    """int8 is what buys the fit, so nothing streams: see the ladder in core/CLAUDE.md. Offloading
    a quantised denoiser puts it in host RAM, where it collides with the conditioner."""
    recipe = recipe_for(_Policy("int8"), keep_precision=H3_KEEP)
    assert recipe.quantize is Quantization.INT8
    assert recipe.keep_precision == H3_KEEP
    assert recipe.denoiser_offload is None and not recipe.use_stream
    assert recipe.vae_offload == LEAF_LEVEL  # idles during denoise, so it is the one that streams


def test_a_large_conditioner_makes_the_denoiser_stream_instead() -> None:
    """MiniMax H3's conditioner is a 32B model that must be co-resident, which the fit ladder does
    not size. The caller says so and the denoiser gives up the card."""
    recipe = recipe_for(_Policy("int8"), keep_precision=H3_KEEP, stream_denoiser=True)
    assert recipe.denoiser_offload == BLOCK_LEVEL and recipe.use_stream


def test_the_tightest_plan_drops_to_leaf_level_and_takes_the_vae_with_it() -> None:
    recipe = recipe_for(_Policy("offload"), keep_precision=H3_KEEP)
    assert recipe.denoiser_offload == LEAF_LEVEL and recipe.vae_offload == LEAF_LEVEL
    assert not recipe.use_stream  # no pinned staging buffer to spare at this size
    assert any("slow" in note for note in recipe.notes)


def test_wont_fit_is_carried_through_rather_than_turned_into_a_recipe() -> None:
    recipe = recipe_for(_Policy("wont-fit"))
    assert recipe.plan == "wont-fit" and not recipe.quantizes
    assert recipe.denoiser_offload is None


def test_a_policy_with_no_estimate_falls_back_to_resident() -> None:
    assert recipe_for(_Policy(None)).plan == "resident"


def test_describe_names_the_plan_for_the_run_log() -> None:
    line = describe(recipe_for(_Policy("int8"), keep_precision=H3_KEEP))
    assert "plan=int8" in line and "resident" in line


# --- split residency ------------------------------------------------------------------------------

GB = 1_000_000_000


def test_a_model_that_fits_ram_places_nothing_on_the_card() -> None:
    """The split costs VRAM the render wants, so it only happens when RAM leaves no choice."""
    assert blocks_to_place(
        model_bytes=20 * GB, block_bytes=GB, free_ram_bytes=64 * GB, ram_headroom_bytes=6 * GB
    ) == 0


def test_only_the_overflow_moves() -> None:
    """66 GB of weights plus 6 GB of headroom against 56 GB free: 16 GB has nowhere to sit."""
    placed = blocks_to_place(
        model_bytes=66 * GB, block_bytes=2 * GB, free_ram_bytes=56 * GB, ram_headroom_bytes=6 * GB
    )
    assert placed == 8  # 16 GB over, in 2 GB blocks - not the 33 the card could have held


def test_a_partial_block_still_costs_a_whole_one() -> None:
    assert blocks_to_place(
        model_bytes=61 * GB, block_bytes=2 * GB, free_ram_bytes=60 * GB, ram_headroom_bytes=6 * GB
    ) == 4  # 7 GB over, rounded up


def test_the_card_reserves_room_for_the_render_before_holding_any_blocks() -> None:
    fits = blocks_that_fit(free_vram_bytes=30 * GB, block_bytes=2 * GB, reserve_bytes=10 * GB)
    assert fits == 10
    assert blocks_that_fit(free_vram_bytes=8 * GB, block_bytes=2 * GB, reserve_bytes=10 * GB) == 0


# --- full precision, for a machine with the RAM ---------------------------------------------------


def test_precision_only_streams_instead_of_quantising() -> None:
    """The numerics gate needs weights nothing has rounded. The fit ladder never picks this, so the
    caller asks for it and pays for it in host RAM."""
    recipe = recipe_for(_Policy("int8"), keep_precision=H3_KEEP, quantize=False)
    assert not recipe.quantizes
    assert recipe.denoiser_offload == BLOCK_LEVEL and recipe.vae_offload == LEAF_LEVEL


def test_precision_only_does_not_override_a_plan_that_already_fits() -> None:
    assert recipe_for(_Policy("resident"), quantize=False).denoiser_offload is None


# --- the prepared cache ---------------------------------------------------------------------------


@pytest.fixture()
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    return tmp_path


def _key(source: Path, **flags: Any) -> PreparedKey:
    return PreparedKey(
        source=source, plan_version="h3-v1", quantization="int8", flags=flags or {"adaln": False}
    )


def _source(root: Path) -> Path:
    path = root / "minimax_h3_fl2va_bf16.safetensors"
    path.write_bytes(b"weights")
    return path


def test_a_miss_then_a_hit(models_root: Path) -> None:
    key = _key(_source(models_root))
    assert lookup(key) is None

    published = publish(key, lambda staging: (staging / "model.safetensors").write_bytes(b"q"))

    assert lookup(key) == published
    assert (published / "model.safetensors").read_bytes() == b"q"


def test_the_cache_lives_outside_the_catalog_scan(models_root: Path) -> None:
    """A prepared artifact must not appear in the model picker as an installed checkpoint."""
    assert prepared_dir(_key(_source(models_root))).parent.name.startswith(".")


def test_changing_a_model_flag_changes_the_identity(models_root: Path) -> None:
    """Switching between the pruned and unpruned paths must not serve the other one's bytes."""
    source = _source(models_root)
    assert prepared_dir(_key(source, adaln=False)) != prepared_dir(_key(source, adaln=True))


def test_the_exclusion_list_is_part_of_the_identity(models_root: Path) -> None:
    """Not redundant with the flags: a structural transform can change which layers are safe to
    quantise, so two artifacts can share a flag set and still hold differently rounded weights."""
    source = _source(models_root)
    wider = PreparedKey(
        source=source, plan_version="h3-v1", quantization="int8",
        flags={"adaln": True}, keep_precision=("proj_in", "adaln_proj"),
    )
    narrower = PreparedKey(
        source=source, plan_version="h3-v1", quantization="int8",
        flags={"adaln": True}, keep_precision=("proj_in",),
    )
    assert prepared_dir(wider) != prepared_dir(narrower)


def test_changing_the_plan_version_changes_the_identity(models_root: Path) -> None:
    source = _source(models_root)
    a = PreparedKey(source=source, plan_version="h3-v1", quantization="int8", flags={})
    b = PreparedKey(source=source, plan_version="h3-v2", quantization="int8", flags={})
    assert prepared_dir(a) != prepared_dir(b)


def test_a_rewritten_source_invalidates_the_artifact(models_root: Path) -> None:
    source = _source(models_root)
    before = prepared_dir(_key(source))
    source.write_bytes(b"different length entirely")
    assert prepared_dir(_key(source)) != before


def test_a_crashed_prepare_leaves_nothing_that_looks_finished(models_root: Path) -> None:
    key = _key(_source(models_root))

    def explode(staging: Path) -> None:
        (staging / "half-written.safetensors").write_bytes(b"partial")
        raise RuntimeError("out of disk")

    with pytest.raises(RuntimeError):
        publish(key, explode)

    assert lookup(key) is None
    assert not list(prepared_dir(key).parent.glob("*.part"))


def test_prune_clears_everything_it_is_not_told_to_keep(models_root: Path) -> None:
    source = _source(models_root)
    keep = publish(_key(source, adaln=False), lambda s: None)
    publish(_key(source, adaln=True), lambda s: None)

    assert prune(keep={keep}) == 1
    assert lookup(_key(source, adaln=False)) == keep
    assert lookup(_key(source, adaln=True)) is None
