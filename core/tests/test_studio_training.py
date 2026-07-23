"""LoRA training: dataset/run CRUD, the orchestrator's stdout parsing, and - the real gate - that a
PEFT adapter's keys are the ones the existing LoRA fuser reads (so the train->load loop closes).
"""

from __future__ import annotations

import sqlite3

import pytest
from inline_core.models import lora
from inline_core.studio import training_store as ts
from inline_core.studio.schema import apply_schema
from inline_core.studio.training import _parse_json_line, _safe


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'P', 0, 0)")
    for i in range(3):
        c.execute(
            "INSERT INTO assets (id, project_id, name, file_path, kind, created_at) "
            "VALUES (?, 'p', ?, ?, 'image', 0)",
            (f"a{i}", f"x{i}.png", f"assets/x{i}.png"),
        )
    return c


def test_dataset_and_item_crud(conn: sqlite3.Connection) -> None:
    ds = ts.create_dataset(conn, "Hero", "ohwx")
    assert ds["triggerWord"] == "ohwx"
    # add_items dedups and assigns sequential positions.
    items = ts.add_items(conn, ds["id"], ["a0", "a1", "a2", "a0"])
    assert [it["position"] for it in items] == [0, 1, 2]
    updated = ts.set_caption(conn, items[0]["id"], "a portrait")
    assert updated["caption"] == "a portrait"
    ts.remove_item(conn, items[1]["id"])
    assert len(ts.list_items(conn, ds["id"])) == 2


def test_run_create_and_update(conn: sqlite3.Connection) -> None:
    ds = ts.create_dataset(conn, "Hero", "")
    run = ts.create_run(conn, ds["id"], "Hero LoRA", {"rank": 8, "steps": 1200, "gpuIds": [0, 1]})
    assert run["status"] == "queued"
    assert run["hyperparams"]["rank"] == 8
    assert run["totalSteps"] == 1200
    assert run["hyperparams"]["gpuIds"] == [0, 1]

    patched = ts.update_run(
        conn, run["id"], {"status": "training", "progressFraction": 0.5, "step": 600}
    )
    assert patched["status"] == "training"
    assert patched["progressFraction"] == 0.5
    assert patched["step"] == 600


def test_parse_json_line_ignores_noise() -> None:
    assert _parse_json_line('{"type": "progress", "step": 5}') == {"type": "progress", "step": 5}
    assert _parse_json_line("loading weights...") is None
    assert _parse_json_line("") is None
    assert _parse_json_line("[1,2,3]") is None  # not an object


def test_safe_filename() -> None:
    assert _safe("My Hero! v2") == "My_Hero_v2"
    assert _safe("///") == "lora"


def test_peft_adapter_keys_are_fuser_compatible() -> None:
    """A trained LoRA is saved as a PEFT state dict; its keys must be exactly what the fuser groups
    + matches, or the LoRA would load with 'unmatched key' and never apply."""
    stem = "base_model.model.transformer_blocks.0.attn.to_q"
    state = {f"{stem}.lora_A.weight": object(), f"{stem}.lora_B.weight": object()}

    pairs, _alphas = lora._group(state)
    assert list(pairs) == [stem]  # down/up recognized and paired

    # The fuser strips the PEFT `base_model.model.` prefix, yielding the real module path.
    assert "transformer_blocks.0.attn.to_q" in lora._candidates(stem)
