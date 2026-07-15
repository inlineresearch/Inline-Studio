from __future__ import annotations

import pytest
from inline_core.errors import CycleError
from inline_core.graph.topo import topo_sort, upstream_closure

# a -> b -> d, a -> c -> d  (edges point to dependencies: d depends on b and c)
_EDGES = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}


def _edges(node_id: str) -> list[str]:
    return _EDGES[node_id]


def test_upstream_closure_is_inclusive() -> None:
    assert upstream_closure("d", _edges) == {"a", "b", "c", "d"}
    assert upstream_closure("b", _edges) == {"a", "b"}


def test_topo_sort_orders_dependencies_first() -> None:
    order = topo_sort(["a", "b", "c", "d"], _edges)
    assert order.index("a") < order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_topo_sort_detects_cycle() -> None:
    cyclic = {"x": ["y"], "y": ["x"]}
    with pytest.raises(CycleError):
        topo_sort(["x", "y"], lambda n: cyclic[n])
