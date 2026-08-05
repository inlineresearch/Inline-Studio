"""The four H3 nodes: their descriptors, how a request resolves, and what the provider offers."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from inline_core.errors import ComponentError  # noqa: E402
from inline_core.graph.schema import PortKind  # noqa: E402
from inline_core.media import MediaKind  # noqa: E402
from inline_core.models.minimaxh3 import requirements as reqs  # noqa: E402
from inline_core.models.minimaxh3.provider import MiniMaxH3Provider  # noqa: E402
from inline_core.models.minimaxh3.runner import (  # noqa: E402
    DESCRIPTORS,
    GRID,
    VARIANTS,
    build_request,
    call_kwargs,
)

BY_TYPE = {v.node_type: v for v in VARIANTS}
T2V = "minimax/h3-text-to-video"
I2V = "minimax/h3-image-to-video"
FLF = "minimax/h3-first-last-frame"
REF = "minimax/h3-reference-to-video"


def _defaults(node_type: str) -> dict[str, Any]:
    return DESCRIPTORS[node_type].defaults()


def _prompt(text: str = "a fox in snow") -> dict[str, list[Any]]:
    return {"prompt": [text]}


# --- descriptors ----------------------------------------------------------------------------------


def test_there_are_four_separate_nodes_not_one_with_a_mode() -> None:
    assert set(DESCRIPTORS) == {T2V, I2V, FLF, REF}


@pytest.mark.parametrize("node_type", [T2V, I2V, FLF, REF])
def test_every_node_outputs_video_plus_a_separate_audio_port(node_type: str) -> None:
    descriptor = DESCRIPTORS[node_type]
    assert descriptor.output_kind is MediaKind.VIDEO
    assert [(p.id, p.kind) for p in descriptor.outputs] == [
        ("video", PortKind.VIDEO),
        ("audio", PortKind.AUDIO),
    ]


#: On every Core node, so a load/* subnode can override the dropdowns. Not part of what a node is
#: *for*, which is what the media ports below say.
COMPONENTS = ["model", "vae", "text_encoder", "lora"]


def test_the_inputs_are_what_each_node_is_for() -> None:
    def media(node_type: str) -> list[str]:
        return [p.id for p in DESCRIPTORS[node_type].inputs if p.id not in COMPONENTS]

    assert media(T2V) == ["prompt"]
    assert media(I2V) == ["prompt", "image"]
    assert media(FLF) == ["prompt", "image", "last_image"]
    assert media(REF) == ["prompt", "references", "video", "audio"]


@pytest.mark.parametrize("node_type", [T2V, I2V, FLF, REF])
def test_every_node_carries_the_component_handles(node_type: str) -> None:
    """They were missing at first, which left H3 the only model family on the canvas with no way to
    wire a checkpoint in. Optional, so nothing is required to draw a connection."""
    ports = {p.id: p for p in DESCRIPTORS[node_type].inputs}
    for name in COMPONENTS:
        assert name in ports, f"{node_type} has no {name} handle"
        assert not ports[name].required
    # The LoRA handle is on every variant, including ref2va: the two partitions are structurally
    # identical, so an adapter trained on one loads on the other.
    assert ports["lora"].kind is PortKind.LORA


def test_first_and_last_frame_are_both_optional() -> None:
    """The partition supports either alone or both, so neither may be required."""
    ports = {p.id: p for p in DESCRIPTORS[FLF].inputs}
    assert not ports["image"].required and not ports["last_image"].required


def test_references_is_a_list_port_so_wiring_order_survives() -> None:
    ports = {p.id: p for p in DESCRIPTORS[REF].inputs}
    assert ports["references"].kind is PortKind.IMAGE_LIST


@pytest.mark.parametrize("node_type", [T2V, I2V, FLF, REF])
def test_no_guidance_and_no_negative_prompt_exist(node_type: str) -> None:
    """The checkpoints are guidance-distilled, so these must be absent rather than ignored."""
    keys = {p.key for p in DESCRIPTORS[node_type].params}
    assert not keys & {"guidance", "guidance_scale", "cfg", "negative_prompt"}


@pytest.mark.parametrize("node_type", [T2V, I2V, FLF, REF])
def test_fps_is_not_editable(node_type: str) -> None:
    """A model constant. Editing it only desyncs it from the 17n+5 frame grid."""
    assert "fps" not in {p.key for p in DESCRIPTORS[node_type].params}


def test_only_the_reference_node_offers_reference_detail() -> None:
    assert "ref_image_size" in {p.key for p in DESCRIPTORS[REF].params}
    assert "ref_image_size" not in {p.key for p in DESCRIPTORS[T2V].params}


def test_the_default_duration_is_a_real_grid_point() -> None:
    duration = next(p for p in DESCRIPTORS[T2V].params if p.key == "duration")
    frames = round(float(duration.default) * GRID.fps)
    assert (frames - GRID.offset) % GRID.grid == 0


# --- building a request ---------------------------------------------------------------------------


def test_a_duration_snaps_onto_the_frame_grid() -> None:
    params = {**_defaults(T2V), "duration": 14.9}
    request = build_request(BY_TYPE[T2V], params, _prompt())
    assert request.num_frames == 345 and request.seconds == pytest.approx(14.375)


def test_a_canvas_snaps_to_the_multiple() -> None:
    params = {**_defaults(T2V), "width": 1000, "height": 500}
    request = build_request(BY_TYPE[T2V], params, _prompt())
    assert (request.width, request.height) == (992, 512)


def test_a_missing_prompt_is_refused_by_name() -> None:
    with pytest.raises(ComponentError, match="needs a prompt"):
        build_request(BY_TYPE[T2V], _defaults(T2V), {})


def test_the_reference_node_needs_at_least_one_reference() -> None:
    with pytest.raises(ComponentError, match="at least one reference"):
        build_request(BY_TYPE[REF], _defaults(REF), _prompt())


def test_too_many_references_is_refused_before_any_load() -> None:
    inputs = {**_prompt(), "references": [f"img{i}" for i in range(10)]}
    with pytest.raises(ComponentError, match="at most 9"):
        build_request(BY_TYPE[REF], _defaults(REF), inputs)


def test_reference_wiring_order_reaches_the_call(tmp_path: Any) -> None:
    """Order is the numbering the prompt addresses, and the call must carry the type the blocks
    accept.

    An earlier version of this test asserted `r.value` on our own `Reference` objects, so it passed
    while the node handed the vendored blocks a type they reject outright. It was asserting the bug.
    The contract is the vendored `MiniMaxH3Reference`, one medium each, in wiring order.
    """
    pytest.importorskip("PIL")
    from PIL import Image

    from inline_core.models.minimaxh3.vendor import MiniMaxH3Reference
    from inline_core.takes import AssetRef

    lead, dog = tmp_path / "lead.png", tmp_path / "dog.png"
    for path in (lead, dog):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(path)
    # A real, decodable clip: the vendored reference decodes audio when it is built, so a stub
    # file would only prove that a stub file fails.
    av = pytest.importorskip("av")
    voice = tmp_path / "voice.wav"
    with av.open(str(voice), "w") as out:
        stream = out.add_stream("pcm_s16le", rate=16000, layout="mono")
        frame = av.AudioFrame(format="s16", layout="mono", samples=1600)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = 16000
        out.mux(stream.encode(frame))
        out.mux(stream.encode(None))

    asset = lambda path: AssetRef(ref="path", path=str(path))  # noqa: E731 - reads better inline
    inputs = {
        **_prompt(),
        "references": [asset(lead), asset(dog)],
        "audio": [asset(voice)],
    }
    request = build_request(BY_TYPE[REF], _defaults(REF), inputs)
    call = call_kwargs(request, BY_TYPE[REF], inputs)

    assert all(isinstance(r, MiniMaxH3Reference) for r in call["references"])
    assert [r.image is not None for r in call["references"]] == [True, True, False]
    # The vendored dataclass decodes at construction: no H3 block opens a media file, so what
    # arrives is a waveform and the rate that came with it, not a path.
    audio_ref = call["references"][2]
    assert audio_ref.image is None and audio_ref.video is None
    assert audio_ref.sample_rate == 16000
    assert audio_ref.audio is not None and audio_ref.audio.ndim == 2


def test_a_text_to_video_call_carries_no_keyframes() -> None:
    request = build_request(BY_TYPE[T2V], _defaults(T2V), _prompt())
    call = call_kwargs(request, BY_TYPE[T2V], _prompt())
    assert "image" not in call and "last_image" not in call and "references" not in call
    assert call["output_type"] == "pil"
    assert set(call) >= {"prompt", "num_frames", "height", "width", "num_inference_steps"}


def test_the_seed_is_resolved_to_a_concrete_value() -> None:
    """-1 means random, but the take has to record what was actually used."""
    request = build_request(BY_TYPE[T2V], {**_defaults(T2V), "seed": -1}, _prompt())
    assert request.seed >= 0
    fixed = build_request(BY_TYPE[T2V], {**_defaults(T2V), "seed": 99}, _prompt())
    assert fixed.seed == 99


# --- recognising checkpoints ----------------------------------------------------------------------


def _fake_checkpoint(path: Path, keys: dict[str, list[int]]) -> Path:
    header = {
        name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in keys.items()
    }
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


_H3_PROBE = {"blocks.0.attn.qkv_proj.weight": [21504, 5376]}


@pytest.fixture()
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    reqs._inspect_cached.cache_clear()
    (tmp_path / "diffusion_models").mkdir(parents=True)
    return tmp_path


def test_an_h3_checkpoint_is_recognised_by_header_not_name(models_root: Path) -> None:
    path = _fake_checkpoint(
        models_root / "diffusion_models" / "renamed_by_a_user.safetensors", _H3_PROBE
    )
    assert reqs.inspect_file(path).usable


def test_another_architecture_is_not_offered(models_root: Path) -> None:
    path = _fake_checkpoint(
        models_root / "diffusion_models" / "minimax_h3_looking_name.safetensors",
        {"blocks.0.attn.qkv_proj.weight": [9216, 3072]},
    )
    assert not reqs.inspect_file(path).is_h3


def test_the_pruned_build_is_rejected_with_a_reason(models_root: Path) -> None:
    path = _fake_checkpoint(
        models_root / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        {**_H3_PROBE, "adaln_t_table": [1025, 8]},
    )
    candidate = reqs.inspect_file(path)
    assert candidate.is_h3 and not candidate.usable
    assert "rank-8 lookup table" in candidate.reason


def test_the_comfy_int8_build_is_rejected_with_a_reason(models_root: Path) -> None:
    path = _fake_checkpoint(
        models_root / "diffusion_models" / "minimax_h3_fl2va_int8_convrot.safetensors",
        {**_H3_PROBE, "blocks.0.attn.qkv_proj.comfy_quant": [72]},
    )
    candidate = reqs.inspect_file(path)
    assert candidate.is_h3 and not candidate.usable
    assert "only ComfyUI" in candidate.reason


def test_the_picker_offers_the_usable_file_and_explains_the_rest(models_root: Path) -> None:
    _fake_checkpoint(models_root / "diffusion_models" / "good.safetensors", _H3_PROBE)
    _fake_checkpoint(
        models_root / "diffusion_models" / "pruned.safetensors",
        {**_H3_PROBE, "adaln_t_table": [1025, 8]},
    )
    provider = MiniMaxH3Provider("fl2va")

    assert provider.catalog_options("diffusion_models") == ["good.safetensors"]
    assert [r["file"] for r in provider.rejected()] == ["pruned.safetensors"]
    assert provider.catalog_options("loras") is None  # not ours to filter


def test_a_picked_transformer_wins_over_the_default_name(models_root: Path) -> None:
    """The dropdown is the recovery path for a renamed checkpoint, so an explicit pick beats the
    expected filename even when that filename is sitting right there."""
    default = _fake_checkpoint(
        models_root / "diffusion_models" / reqs.FL2VA_FILE, _H3_PROBE
    )
    picked = _fake_checkpoint(
        models_root / "diffusion_models" / "renamed_by_a_user.safetensors", _H3_PROBE
    )

    assert reqs.resolve_transformer("fl2va") == default
    assert reqs.resolve_transformer("fl2va", "renamed_by_a_user.safetensors") == picked


def test_a_pick_that_is_not_there_falls_back_rather_than_failing(models_root: Path) -> None:
    default = _fake_checkpoint(
        models_root / "diffusion_models" / reqs.FL2VA_FILE, _H3_PROBE
    )
    assert reqs.resolve_transformer("fl2va", "no_such_file.safetensors") == default
    assert reqs.resolve_transformer("fl2va", "") == default
    assert reqs.resolve_transformer("fl2va", None) == default


def test_a_picked_transformer_is_what_gets_sized(models_root: Path) -> None:
    """A footprint taken from the default name would be zero once the pick is a renamed file, and a
    zero footprint makes the fit ladder wave through a model that cannot load."""
    picked = _fake_checkpoint(
        models_root / "diffusion_models" / "renamed_by_a_user.safetensors", _H3_PROBE
    )
    assert reqs.footprint_bytes("fl2va", factorised=False)["diffusion_bytes"] == 0

    raw = reqs.footprint_bytes("fl2va", factorised=False, transformer=picked)["diffusion_bytes"]
    assert raw == picked.stat().st_size > 0


def test_the_factorised_share_comes_off_whichever_file_was_picked(models_root: Path) -> None:
    """The two knobs compose: the pick decides *which* file is sized, ``factorised`` decides how
    much of it the policy is told to place."""
    picked = _fake_checkpoint(
        models_root / "diffusion_models" / "renamed_by_a_user.safetensors", _H3_PROBE
    )
    raw = reqs.footprint_bytes("fl2va", factorised=False, transformer=picked)["diffusion_bytes"]
    placed = reqs.footprint_bytes("fl2va", factorised=True, transformer=picked)["diffusion_bytes"]
    assert placed == int(raw * (1 - reqs.ADALN_SHARE)) < raw


def test_header_reads_are_cached_against_size_and_mtime(models_root: Path) -> None:
    path = _fake_checkpoint(models_root / "diffusion_models" / "a.safetensors", _H3_PROBE)
    assert reqs.inspect_file(path).usable

    _fake_checkpoint(path, {"something.else": [4, 4]})
    # Same path, different bytes: the cache key includes size, so this is re-read rather than stale.
    assert not reqs.inspect_file(path).is_h3


# --- the provider ---------------------------------------------------------------------------------


def test_the_reference_node_requires_the_other_partition(models_root: Path) -> None:
    fl2va = {c.id: c for c in MiniMaxH3Provider("fl2va").components()}
    ref2va = {c.id: c for c in MiniMaxH3Provider("ref2va").components()}
    assert fl2va["h3-ref2va"].optional and not fl2va["h3-fl2va"].optional
    assert ref2va["h3-fl2va"].optional and not ref2va["h3-ref2va"].optional


def test_the_folder_components_declare_a_repo_folder(models_root: Path) -> None:
    by_id = {c.id: c for c in MiniMaxH3Provider().components()}
    assert by_id["h3-text-encoder"].is_folder and by_id["h3-processor"].is_folder
    assert not by_id["h3-fl2va"].is_folder and by_id["h3-fl2va"].repo_file.endswith(".safetensors")


def test_provenance_survives_a_rename(models_root: Path) -> None:
    """The two partitions are indistinguishable by inspection, so this is the only record."""
    renamed = _fake_checkpoint(
        models_root / "diffusion_models" / "my_ref_model.safetensors", _H3_PROBE
    )
    assert reqs.resolve_transformer("ref2va") is None

    reqs.record_provenance("ref2va", renamed.name)

    assert reqs.resolve_transformer("ref2va") == renamed


def test_each_blockset_gets_its_denoiser_under_the_name_it_declares() -> None:
    """Ref2VA calls it `transformer_ref`, FL2VA calls it `transformer`, and `update_components`
    only warns about a name it does not know. Pass the wrong one and the pipeline ends up with no
    denoiser at all, which surfaces much later as a missing attribute part-way into a render."""
    pytest.importorskip("torch")
    from inline_core.models.minimaxh3.pipeline import _denoiser_name
    from inline_core.models.minimaxh3.vendor import MiniMaxH3Blocks, MiniMaxH3Ref2VABlocks

    assert _denoiser_name(MiniMaxH3Blocks()) == "transformer"
    assert _denoiser_name(MiniMaxH3Ref2VABlocks()) == "transformer_ref"
