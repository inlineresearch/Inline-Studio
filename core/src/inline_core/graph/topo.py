"""Pure graph traversal: the upstream closure and a stable topological sort."""

from __future__ import annotations

from collections.abc import Callable

from ..errors import CycleError

Edges = Callable[[str], list[str]]


def upstream_closure(target: str, edges: Edges) -> set[str]:
    """Every node reachable upstream of `target`, inclusive."""
    seen: set[str] = set()
    stack = [target]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for up in edges(node_id):
            if up not in seen:
                stack.append(up)
    return seen


def topo_sort(ids: list[str], edges: Edges) -> list[str]:
    """Kahn sort over `ids` (dependencies first). Raises CycleError on a residual cycle."""
    id_set = set(ids)
    indegree = dict.fromkeys(ids, 0)
    dependents: dict[str, list[str]] = {i: [] for i in ids}
    for i in ids:
        for up in edges(i):
            if up not in id_set:
                continue
            indegree[i] += 1
            dependents[up].append(i)
    queue = [i for i in ids if indegree[i] == 0]
    order: list[str] = []
    while queue:
        node_id = queue.pop(0)
        order.append(node_id)
        for dep in dependents[node_id]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)
    if len(order) != len(ids):
        raise CycleError("This graph has a cycle.")
    return order
