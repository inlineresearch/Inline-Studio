"""Train LoRA's base checkpoint, which no param on the node names.

The engine resolves it from the architecture at run time, so an exported training graph listed the
character encoders and left its largest download for the reader to work out.
"""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.models.trainingreqs import TRAINING_NODES, TrainingBaseProvider


def _files(arch: str, base_mode: str = "") -> list[tuple[str, str]]:
    got = TrainingBaseProvider().components({"hyperparams": {"arch": arch, "baseMode": base_mode}})
    return [(c.category, c.filename) for c in got]


def test_it_answers_for_the_train_lora_node() -> None:
    assert TRAINING_NODES == ("train/lora",)


def test_krea2_follows_the_base_the_settings_picked() -> None:
    """RAW is the fine-tuning build; Turbo only when the run adds the de-distillation adapter."""
    assert ("diffusion_models", "krea2_raw_bf16.safetensors") in _files("krea2", "raw")
    assert ("diffusion_models", "krea2_turbo_bf16.safetensors") in _files("krea2", "turbo_adapter")


def test_flux2_asks_for_the_base_build_not_the_distilled_one() -> None:
    """The trainer refuses a step-distilled checkpoint, so listing the generation default would
    publish a workflow whose own engine rejects the file it named."""
    names = [f for _c, f in _files("flux2", "raw")]
    assert "flux-2-klein-base-4b.safetensors" in names
    assert "flux-2-klein-4b.safetensors" not in names


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("z-image", "z_image_bf16.safetensors"),
        ("ltx-2-5", "ltx-2.5-22b-dev-transformer-bf16.safetensors"),
        ("minimax-h3", "minimax_h3_fl2va_bf16.safetensors"),
    ],
)
def test_every_trainable_arch_names_its_transformer(arch: str, expected: str) -> None:
    assert ("diffusion_models", expected) in _files(arch)


def test_an_unset_architecture_asks_for_nothing() -> None:
    """A node whose settings have not been opened must not publish a checkpoint nobody chose."""
    assert TrainingBaseProvider().components({}) == []
    assert TrainingBaseProvider().components(None) == []
    assert _files("not-an-arch") == []


def test_suggested_extras_stay_out_of_a_training_pre_flight() -> None:
    """Krea 2's depth control-LoRA is offered for generation; a training run never loads it."""
    assert all("control" not in f for _c, f in _files("krea2", "raw"))


def test_the_recipe_puts_them_on_the_training_node() -> None:
    """On the node that needs it, the way a core node carries its own models."""
    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_trainer(conn, 0, 0)
    hyper = {"arch": "krea2", "baseMode": "raw"}
    mb.update_item(conn, node["id"], {"data": {"hyperparams": hyper}})

    provider = TrainingBaseProvider()
    studio_recipe.set_model_resolver(
        lambda node_type, params=None: [
            (c.filename, c.category)
            for c in (provider.components(params) if node_type == "train/lora" else [])
            if not c.optional
        ]
    )
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_model_resolver(None)

    data = built["graph"]["items"][0]["data"]
    assert data["hyperparams"]["arch"] == "krea2"
    names = [m["name"] for m in data["models"]]
    assert "krea2_raw_bf16.safetensors" in names
