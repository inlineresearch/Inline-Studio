from __future__ import annotations

import pytest
from inline_core.errors import GraphValidationError
from inline_core.graph.schema import Edge, PortKind, parse_graph, port_satisfies


def test_port_satisfies_exact_and_image_into_list() -> None:
    assert port_satisfies(PortKind.TEXT, PortKind.TEXT)
    assert port_satisfies(PortKind.IMAGE, PortKind.IMAGE_LIST)
    assert not port_satisfies(PortKind.IMAGE_LIST, PortKind.IMAGE)
    assert not port_satisfies(PortKind.VIDEO, PortKind.IMAGE)


def test_parse_graph_normalizes_single_and_list_edges() -> None:
    graph = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "p1", "type": "input/text", "params": {"text": "hi"}},
                {
                    "id": "m1",
                    "type": "fake",
                    "inputs": {"prompt": {"from": "p1", "output": "text"}},
                },
                {
                    "id": "m2",
                    "type": "fake",
                    "inputs": {"imgs": [{"from": "m1", "output": "image"}]},
                },
            ],
        }
    )
    assert graph.node("m1").inputs["prompt"] == [Edge("p1", "text")]
    assert graph.node("m2").inputs["imgs"] == [Edge("m1", "image")]
    assert graph.input_sources("m1") == ["p1"]


def test_parse_graph_rejects_bad_version_and_duplicates() -> None:
    with pytest.raises(GraphValidationError):
        parse_graph({"schemaVersion": 2, "nodes": []})
    with pytest.raises(GraphValidationError):
        parse_graph(
            {
                "schemaVersion": 1,
                "nodes": [
                    {"id": "a", "type": "input/text"},
                    {"id": "a", "type": "input/text"},
                ],
            }
        )
