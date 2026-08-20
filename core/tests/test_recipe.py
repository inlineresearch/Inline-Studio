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


def test_recipe_records_what_a_node_resolves_to(tmp_path) -> None:
    """A picker left alone stores nothing, so a recipe built from the item alone names no weight
    and cannot rebuild the same image anywhere else."""
    import sqlite3

    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_core_node(conn, "load/diffusion-model", 0, 0)

    studio_recipe.set_param_resolver(lambda _t: {"file": "flux-2-klein-9b.safetensors"})
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_param_resolver(None)

    params = built["graph"]["items"][0]["data"]["core"]["params"]
    assert params["file"]["value"] == "flux-2-klein-9b.safetensors"


def test_a_picked_file_beats_what_the_node_would_resolve(tmp_path) -> None:
    import sqlite3

    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_core_node(conn, "load/diffusion-model", 0, 0)
    mb.update_item(
        conn,
        node["id"],
        {
            "data": {
                "core": {"type": "load/diffusion-model", "params": {"file": "picked.safetensors"}}
            }
        },
    )

    studio_recipe.set_param_resolver(lambda _t: {"file": "auto.safetensors"})
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_param_resolver(None)

    params = built["graph"]["items"][0]["data"]["core"]["params"]
    assert params["file"]["value"] == "picked.safetensors"


def test_a_recipe_survives_a_resolver_that_raises(tmp_path) -> None:
    """Recipe data is metadata on a finished render; it must never fail the render."""
    import sqlite3

    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_core_node(conn, "load/diffusion-model", 0, 0)

    def boom(_t: str) -> dict:
        raise RuntimeError("registry is gone")

    studio_recipe.set_param_resolver(boom)
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_param_resolver(None)

    assert built["graph"]["items"][0]["data"]["core"]["params"] == {}


def test_resolved_params_agree_with_what_the_node_serves(tmp_path, monkeypatch) -> None:
    """The recipe and the node face must name the same file. Reading it back off the served
    descriptor is what keeps them from drifting: an empty pick means the first option."""
    from inline_core.graph.descriptor import NodeDescriptor, ParamField, Widget
    from inline_core.models.catalog import ModelCatalog
    from inline_core.server.serialize import descriptor_json, resolved_params

    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path))
    catalog = ModelCatalog(tmp_path)
    catalog.ensure_dirs()
    (tmp_path / "diffusion_models" / "a.safetensors").write_bytes(b"x")
    (tmp_path / "diffusion_models" / "b.safetensors").write_bytes(b"x")
    catalog.rescan()

    descriptor = NodeDescriptor(
        type="load/diffusion-model",
        title="Load Diffusion Model",
        category="Loaders",
        params=(
            ParamField("file", "File", Widget.SELECT, "", options_from="diffusion_models"),
            ParamField("steps", "Steps", Widget.NUMBER, 8),
        ),
    )

    params = resolved_params(descriptor, catalog)
    served = {f["key"]: f for f in descriptor_json(descriptor, catalog)["params"]}

    assert params["file"] == served["file"]["options"][0]["value"]
    assert params["steps"] == 8, "a plain default is recorded too, not only file picks"


def test_a_training_node_keeps_its_settings_but_not_its_bindings() -> None:
    """Hyperparams describe the run and travel. A dataset id and a run id are rows in this
    project's database, so they name nothing in the project a recipe lands in."""
    from inline_core.studio.recipe import _clean_data

    trainer = _clean_data(
        {
            "type": "train/lora",
            "data": {
                "hyperparams": {"arch": "krea2", "rank": 32, "steps": 1200},
                "datasetId": "d-local",
                "runId": "r-local",
            },
        }
    )
    assert trainer == {"hyperparams": {"arch": "krea2", "rank": 32, "steps": 1200}}

    caption = _clean_data(
        {"type": "train/caption", "data": {"overwrite": True, "datasetId": "d-local"}}
    )
    assert caption == {"overwrite": True, "captioner": ""}

    assert _clean_data({"type": "train/dataset", "data": {"datasetId": "d-local"}}) == {}
    assert _clean_data({"type": "train/loss", "data": {"runId": "r-local"}}) == {}


def test_the_recipe_lists_models_no_param_names() -> None:
    """A character node picks no file, so its encoders are in no param; LTX-2.5's duration head is
    the same. An exported graph listed none of them while refusing to run without them."""
    import sqlite3

    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_core_node(conn, "character/encode", 0, 0)

    studio_recipe.set_model_resolver(
        lambda _t, _p=None: [("face_detection_yunet_2023mar.onnx", "annotators")]
    )
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_model_resolver(None)

    # On the node that needs it, not in a list beside the graph: everything a reader needs is
    # reachable from the node it belongs to.
    models = built["graph"]["items"][0]["data"]["core"]["models"]
    assert [m["name"] for m in models] == ["face_detection_yunet_2023mar.onnx"]
    assert models[0]["directory"] == "annotators"
    assert "models" not in built, "the graph-wide list is gone"


def test_the_model_list_also_carries_the_weights_a_param_picks() -> None:
    """One list is the whole answer, rather than half of it hidden in params."""
    import sqlite3

    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_core_node(conn, "load/vae", 0, 0)
    mb.update_item(
        conn,
        node["id"],
        {"data": {"core": {"type": "load/vae", "params": {"file": "flux2-vae.safetensors"}}}},
    )

    studio_recipe.set_kind_resolver(lambda _t: {"file": "model"})
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_kind_resolver(None)

    core = built["graph"]["items"][0]["data"]["core"]
    assert core["params"]["file"] == {"type": "model", "value": "flux2-vae.safetensors"}
    assert [m["name"] for m in core["models"]] == ["flux2-vae.safetensors"]


def test_a_param_says_what_it_is_rather_than_leaving_it_to_be_guessed() -> None:
    """A bare value forces a reader to guess from the param's name, which is how a control-LoRA
    ends up filed under loras. The kind travels with the value instead."""
    import sqlite3

    from inline_core.studio import moodboard as mb
    from inline_core.studio import recipe as studio_recipe
    from inline_core.studio.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    conn.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p','P',0,0)")
    node = mb.add_core_node(conn, "character/encode", 0, 0)
    mb.update_item(
        conn,
        node["id"],
        {
            "data": {
                "core": {
                    "type": "character/encode",
                    "params": {"name": "Ada", "subject_embedder": "dinov2-base"},
                }
            }
        },
    )

    studio_recipe.set_kind_resolver(
        lambda _t: {"name": "string", "subject_embedder": "model"}
    )
    try:
        built = studio_recipe.build_recipe(conn, node["id"])
    finally:
        studio_recipe.set_kind_resolver(None)

    params = built["graph"]["items"][0]["data"]["core"]["params"]
    assert params["name"] == {"type": "string", "value": "Ada"}
    assert params["subject_embedder"] == {"type": "model", "value": "dinov2-base"}
    # Only the model-kinded one earns a coordinates entry.
    assert [m["name"] for m in built["graph"]["items"][0]["data"]["core"]["models"]] == [
        "dinov2-base"
    ]


def test_the_version_says_which_shape_a_reader_is_holding() -> None:
    from inline_core.studio.recipe import RECIPE_VERSION

    assert RECIPE_VERSION == 2


def test_a_wired_model_param_is_not_exported_as_a_needed_model(tmp_path) -> None:
    """`model`/`vae`/`text_encoder` are params and input ports at once. Wired, the loader wins at
    run time, so exporting the typed value listed the wrong checkpoint beside the right one."""
    from inline_core.studio import recipe as studio_recipe

    store = _store(tmp_path)
    conn = store.conn()
    gen = mb.add_core_node(conn, "krea/krea-2-turbo", 400, 200)
    mb.update_item(conn, gen["id"], {"data": {"core": {
        "type": "krea/krea-2-turbo",
        "params": {"model": "models/diffusion_models/krea2_turbo_bf16.safetensors"},
    }}})
    loader = mb.add_core_node(conn, "load/diffusion-model", 80, 200)
    mb.update_item(conn, loader["id"], {"data": {"core": {
        "type": "load/diffusion-model", "params": {"file": "krea2_raw_bf16.safetensors"},
    }}})
    mb.create_connector(conn, loader["id"], gen["id"], "model", "model")

    studio_recipe.set_kind_resolver(lambda t: {"model": "model", "file": "model"})
    try:
        built = studio_recipe.build_recipe(conn, gen["id"])
    finally:
        studio_recipe.set_kind_resolver(None)

    by_id = {i["id"]: i for i in built["graph"]["items"]}
    assert by_id[gen["id"]]["data"]["core"].get("models", []) == [], "the wire drives it"
    assert [m["name"] for m in by_id[loader["id"]]["data"]["core"]["models"]] == [
        "krea2_raw_bf16.safetensors"
    ], "the loader names the file the run will load"


def test_a_path_shaped_pick_exports_under_its_bare_name(tmp_path) -> None:
    """A legacy full-path pick matched no registry row, so the same file landed twice: once right
    and once with an empty directory and no download link."""
    from inline_core.studio import recipe as studio_recipe

    store = _store(tmp_path)
    conn = store.conn()
    gen = mb.add_core_node(conn, "krea/krea-2-turbo", 400, 200)
    mb.update_item(conn, gen["id"], {"data": {"core": {
        "type": "krea/krea-2-turbo",
        "params": {"model": "models/diffusion_models/krea2_turbo_bf16.safetensors"},
    }}})

    studio_recipe.set_kind_resolver(lambda t: {"model": "model"})
    try:
        built = studio_recipe.build_recipe(conn, gen["id"])
    finally:
        studio_recipe.set_kind_resolver(None)

    names = [m["name"] for m in built["graph"]["items"][0]["data"]["core"]["models"]]
    assert names == ["krea2_turbo_bf16.safetensors"]

