from __future__ import annotations

import pytest
from helpers import make_registry
from inline_core.errors import PortTypeError, UnknownNodeType
from inline_core.graph.schema import parse_graph
from inline_core.graph.validate import validate


def test_valid_graph_passes() -> None:
    registry = make_registry()
    graph = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {"id": "p1", "type": "input/text", "params": {"text": "x"}},
                {
                    "id": "m1",
                    "type": "fake/model",
                    "inputs": {"prompt": {"from": "p1", "output": "text"}},
                },
            ],
        }
    )
    validate(graph, "m1", registry)


def test_missing_required_input_raises() -> None:
    registry = make_registry()
    graph = parse_graph({"schemaVersion": 1, "nodes": [{"id": "m1", "type": "fake/model"}]})
    with pytest.raises(PortTypeError):
        validate(graph, "m1", registry)


def test_unknown_type_raises() -> None:
    registry = make_registry()
    graph = parse_graph({"schemaVersion": 1, "nodes": [{"id": "m1", "type": "nope/nope"}]})
    with pytest.raises(UnknownNodeType):
        validate(graph, "m1", registry)


def test_wrong_kind_raises() -> None:
    registry = make_registry()
    graph = parse_graph(
        {
            "schemaVersion": 1,
            "nodes": [
                {
                    "id": "i1",
                    "type": "input/image",
                    "params": {"asset": {"ref": "path", "path": "/x.png"}},
                },
                {
                    "id": "m1",
                    "type": "fake/model",
                    "inputs": {"prompt": {"from": "i1", "output": "image"}},
                },
            ],
        }
    )
    with pytest.raises(PortTypeError):
        validate(graph, "m1", registry)
