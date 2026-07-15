"""The graph executor: lazily run a target's closure, reuse cached nodes, stream run events.

It orchestrates cheap work inline. A model node's runner submits the denoise to the batched sampler
(see sampling/batch.py); the executor never runs the loop itself.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..errors import CancelledError, GraphValidationError, InlineCoreError
from ..runtime.context import ExecutionContext
from ..runtime.progress import CancelledEvent, ErrorEvent, NodeDoneEvent, RunDoneEvent
from ..runtime.run import NodeRuntimeState, RunState, RunStatus, StateTrackingEmitter
from .cache import NodeCache, is_cache_eligible, node_cache_key
from .registry import Registry
from .schema import Graph, Node
from .topo import topo_sort, upstream_closure
from .validate import validate


class Executor:
    def __init__(self, registry: Registry, cache: NodeCache) -> None:
        self._registry = registry
        self._cache = cache

    def run(self, graph: Graph, target: str, ctx: ExecutionContext, state: RunState) -> None:
        """Run the target's closure, updating `state` and forwarding events to ctx.emitter."""
        emitter = StateTrackingEmitter(ctx.emitter, state)
        run_ctx = replace(ctx, emitter=emitter)
        try:
            order = self._plan(graph, target, state)
            state.status = RunStatus.RUNNING
            outputs: dict[str, dict[str, Any]] = {}
            for node_id in order:
                if ctx.cancel.cancelled:
                    raise CancelledError("Run cancelled.")
                self._run_node(graph, node_id, outputs, run_ctx)
            emitter.emit(RunDoneEvent(run_id=ctx.run_id))
        except CancelledError:
            emitter.emit(CancelledEvent(run_id=ctx.run_id))
        except GraphValidationError as error:
            emitter.emit(ErrorEvent(run_id=ctx.run_id, message=str(error), node_id=error.node_id))
        except InlineCoreError as error:
            emitter.emit(ErrorEvent(run_id=ctx.run_id, message=str(error)))

    def _plan(self, graph: Graph, target: str, state: RunState) -> list[str]:
        validate(graph, target, self._registry)
        closure = list(upstream_closure(target, graph.input_sources))
        order = topo_sort(closure, graph.input_sources)
        for node_id in order:
            state.nodes.setdefault(node_id, NodeRuntimeState())
        return order

    def _run_node(
        self,
        graph: Graph,
        node_id: str,
        outputs: dict[str, dict[str, Any]],
        ctx: ExecutionContext,
    ) -> None:
        node = graph.node(node_id)
        runner = self._registry.runner(node.type)
        inputs = self._resolve_inputs(node, outputs)

        key: str | None = None
        if runner.produces_takes and is_cache_eligible(node, self._registry):
            # TODO(phase1): pass real asset content hashes so identity is content-addressed.
            key = node_cache_key(graph, node_id, self._registry, asset_hashes={})
            cached = self._cache.get(key)
            if cached is not None:
                ctx.emitter.emit(
                    NodeDoneEvent(run_id=ctx.run_id, node_id=node_id, cached=True, takes=cached)
                )
                outputs[node_id] = self._take_outputs(node, cached[0] if cached else None)
                return

        result = runner.run(node, inputs, ctx)
        outputs[node_id] = result.outputs
        if result.takes:
            if key is not None:
                self._cache.put(key, result.takes)
            ctx.emitter.emit(
                NodeDoneEvent(run_id=ctx.run_id, node_id=node_id, cached=False, takes=result.takes)
            )

    def _resolve_inputs(
        self, node: Node, outputs: dict[str, dict[str, Any]]
    ) -> dict[str, list[Any]]:
        resolved: dict[str, list[Any]] = {}
        for port_id, edges in node.inputs.items():
            values: list[Any] = []
            for edge in edges:
                upstream = outputs.get(edge.from_node, {})
                if edge.output in upstream:
                    values.append(upstream[edge.output])
            resolved[port_id] = values
        return resolved

    def _take_outputs(self, node: Node, take: Any) -> dict[str, Any]:
        return {port.id: take for port in self._registry.get(node.type).outputs}
