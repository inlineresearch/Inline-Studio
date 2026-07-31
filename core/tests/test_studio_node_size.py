"""Generation nodes open large and portrait; plumbing nodes stay compact.

The size is decided at insert from the node's descriptor rather than by the renderer, because only
the registry knows whether a type renders an image preview.
"""

from __future__ import annotations

import sqlite3

import pytest

from inline_core.studio import moodboard as mb
from inline_core.studio.schema import apply_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_schema(c)
    c.execute("INSERT INTO project (id, name, created_at, updated_at) VALUES ('p', 'P', 0, 0)")
    return c


def test_a_core_node_defaults_to_the_compact_size(conn: sqlite3.Connection) -> None:
    item = mb.add_core_node(conn, "load/vae", 0, 0)
    assert (item["width"], item["height"]) == (200, 120)


def test_the_caller_can_size_it(conn: sqlite3.Connection) -> None:
    item = mb.add_core_node(conn, "alibaba/z-image-turbo", 0, 0, 320, 480)
    assert (item["width"], item["height"]) == (320, 480)
    assert item["height"] > item["width"], "generation nodes open portrait"


def test_generation_types_open_large_and_portrait() -> None:
    """The rule the handler applies: an outputKind means a preview, which means the large card."""
    from inline_core.studio.handlers import COMPACT_NODE_SIZE, GENERATION_NODE_SIZE, core_node_size

    models = [
        {"type": "alibaba/z-image-turbo", "outputKind": "image"},
        {"type": "black-forest-labs/flux-2", "outputKind": "image"},
        {"type": "krea/krea-2-turbo", "outputKind": "image"},
        {"type": "load/vae", "outputKind": None},
    ]
    for node_type in ("alibaba/z-image-turbo", "black-forest-labs/flux-2", "krea/krea-2-turbo"):
        assert core_node_size(models, node_type) == GENERATION_NODE_SIZE

    width, height = GENERATION_NODE_SIZE
    assert height > width, "generation nodes open portrait"
    assert width * height > 5 * COMPACT_NODE_SIZE[0] * COMPACT_NODE_SIZE[1]

    # A loader has no preview, and an unregistered type is assumed compact rather than guessed
    # large: a big empty card on the canvas is worse than a small one.
    assert core_node_size(models, "load/vae") == COMPACT_NODE_SIZE
    assert core_node_size(models, "not/registered") == COMPACT_NODE_SIZE
    assert core_node_size([], "alibaba/z-image-turbo") == COMPACT_NODE_SIZE
