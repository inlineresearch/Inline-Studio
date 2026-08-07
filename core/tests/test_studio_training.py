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


class _Store:
    """The two things `Training` asks a store for."""

    def __init__(self, conn: sqlite3.Connection, folder: object) -> None:
        self._conn, self._folder = conn, folder

    def conn(self) -> sqlite3.Connection:
        return self._conn

    def folder(self) -> object:
        return self._folder


def test_add_from_path_imports_a_folder_with_its_sidecar_captions(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    """The clip case: pointing at a folder beats pushing gigabytes through the browser."""
    from pathlib import Path

    from inline_core.studio.training import Training

    src = Path(str(tmp_path)) / "src"
    src.mkdir()
    (src / "0000.png").write_bytes(b"x")
    (src / "0000.txt").write_text("a red car")
    (src / "0001.mp4").write_bytes(b"x")  # a clip, no caption
    (src / "notes.md").write_text("ignored")  # not media

    project = Path(str(tmp_path)) / "project"
    project.mkdir()
    dataset = ts.create_dataset(conn, "d", "")
    items = Training(_Store(conn, project), events=None).add_from_path(dataset["id"], str(src))

    assert len(items) == 2, "the .md is not media and must not be imported"
    captions = {i["caption"] for i in items}
    assert captions == {"a red car", ""}
    # The files are copied into the project rather than referenced in place.
    assert len(list((project / "assets").iterdir())) == 2


def test_add_from_path_rejects_a_folder_that_is_not_one(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    from pathlib import Path

    from inline_core.studio.training import Training

    dataset = ts.create_dataset(conn, "d", "")
    training = Training(_Store(conn, Path(str(tmp_path))), events=None)
    with pytest.raises(ValueError, match="Not a folder"):
        training.add_from_path(dataset["id"], str(Path(str(tmp_path)) / "nope"))


def test_add_from_path_says_so_when_a_folder_holds_no_media(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    from pathlib import Path

    from inline_core.studio.training import Training

    empty = Path(str(tmp_path)) / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("just captions, no images")
    dataset = ts.create_dataset(conn, "d", "")
    training = Training(_Store(conn, Path(str(tmp_path))), events=None)
    with pytest.raises(ValueError, match="No images or clips"):
        training.add_from_path(dataset["id"], str(empty))


def test_each_new_precache_status_becomes_a_log_line() -> None:
    """Precache progress reaches the UI only as a changing progress status. The trainer subprocess
    installs no logging handler, so a logger.info there is dropped and the pane stays silent for
    the minutes a large dataset takes."""
    from inline_core.studio.training import _progress_log_line

    last = ""
    lines = []
    for status in ("caching latents 5/173", "caching latents 10/173", "caching latents 10/173"):
        line = _progress_log_line({"status": status}, last)
        if line is not None:
            lines.append(line)
            last = status
    # The first two are new phases and get a line; the repeat is suppressed.
    assert lines == ["caching latents 5/173", "caching latents 10/173"]


def test_a_step_with_a_loss_still_wins_over_the_status() -> None:
    from inline_core.studio.training import _progress_log_line

    line = _progress_log_line({"status": "training", "step": 7, "total": 500, "loss": 0.1234}, "")
    assert line == "step 7/500 · loss 0.1234"
