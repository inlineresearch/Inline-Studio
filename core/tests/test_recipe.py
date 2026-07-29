"""The generation recipe: authoring the upstream subgraph + params + prompt for a take, and the PNG
tEXt round-trip that makes an image self-describing."""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.studio import moodboard as mb
from inline_core.studio.recipe import RECIPE_VERSION, build_recipe
from inline_core.studio.store import StudioStore


def _store(tmp_path) -> StudioStore:
    store = StudioStore(tmp_path / "app", tmp_path / "ws")
    store.create_project("Recipe")
    return store


def test_build_recipe_captures_params_prompt_and_upstream_subgraph(tmp_path) -> None:
    store = _store(tmp_path)
    conn = store.conn()
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    mb.update_item(conn, z["id"], {"data": {"core": {"type": "alibaba/z-image-turbo",
                                                     "params": {"steps": 8, "seed": 42}}}})
    prompt = mb.add_prompt(conn, 80, 200)
    mb.update_item(conn, prompt["id"], {"data": {"promptText": "a neon fox"}})
    mb.create_connector(conn, prompt["id"], z["id"], "out", "prompt")

    recipe = build_recipe(conn, z["id"])
    assert recipe["version"] == RECIPE_VERSION and recipe["app"] == "inline-studio"
    assert recipe["target"] == z["id"] and recipe["coreType"] == "alibaba/z-image-turbo"
    assert recipe["params"] == {"steps": 8, "seed": 42}
    assert recipe["prompt"] == "a neon fox"
    ids = {i["id"]: i for i in recipe["graph"]["items"]}
    assert set(ids) == {z["id"], prompt["id"]}
    assert ids[prompt["id"]]["data"] == {"promptText": "a neon fox"}
    # The take history of an upstream node must NOT be embedded (no recursive bloat).
    assert "outputs" not in ids[z["id"]]["data"]["core"]
    assert len(recipe["graph"]["connectors"]) == 1


def test_recipe_omits_disconnected_nodes(tmp_path) -> None:
    store = _store(tmp_path)
    conn = store.conn()
    z = mb.add_core_node(conn, "alibaba/z-image-turbo", 400, 200)
    mb.add_prompt(conn, 80, 500)  # a stray prompt, not wired to z
    recipe = build_recipe(conn, z["id"])
    assert [i["id"] for i in recipe["graph"]["items"]] == [z["id"]]


def test_build_recipe_captures_a_fal_gen_node(tmp_path) -> None:
    """A fal gen node is a frame; its model + params live in the frames table, so the recipe must
    look them up (not read the item's empty data) - that is what lets a shared fal PNG rebuild."""
    store = _store(tmp_path)
    conn = store.conn()
    gen = mb.add_gen_node(
        conn, "fal-ai/flux/dev", 400, 200, kind="image", params={"num_images": 2}, title="Flux"
    )
    prompt = mb.add_prompt(conn, 80, 200)
    mb.update_item(conn, prompt["id"], {"data": {"promptText": "a fox"}})
    mb.create_connector(conn, prompt["id"], gen["id"], "out", "prompt")

    recipe = build_recipe(conn, gen["id"])
    assert recipe["prompt"] == "a fox"
    ids = {i["id"]: i for i in recipe["graph"]["items"]}
    fal = ids[gen["id"]]["data"]["fal"]
    assert fal["modelId"] == "fal-ai/flux/dev"
    assert fal["params"] == {"num_images": 2}


def test_png_recipe_round_trip(tmp_path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    from inline_core.studio.image_meta import RECIPE_KEY, embed_recipe_png, read_recipe_png

    src = tmp_path / "src.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src)
    recipe = {"version": 1, "app": "inline-studio", "prompt": "hi", "graph": {"items": []}}
    dst = tmp_path / "out.png"
    embed_recipe_png(src, dst, recipe)

    assert read_recipe_png(dst) == recipe
    # A foreign PNG carries no recipe.
    assert read_recipe_png(src) is None
    # And the chunk really is in the file's text metadata.
    with Image.open(dst) as im:
        assert RECIPE_KEY in im.text


def test_read_recipe_png_tolerates_a_missing_or_bad_file(tmp_path) -> None:
    from inline_core.studio.image_meta import read_recipe_png

    assert read_recipe_png(tmp_path / "nope.png") is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    assert read_recipe_png(bad) is None
    assert read_recipe_png(Path(bad)) is None
