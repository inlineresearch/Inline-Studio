/**
 * Graph-level selection helpers for the canvas: treat a connected set of nodes (joined by
 * connectors) as one runnable graph. Selecting any node selects the whole graph; the graph's Run
 * control lives on its **output node** - the runnable node with nothing runnable downstream of it.
 *
 * Pure + framework-free so it unit-tests without React Flow. Connectors are directional
 * (`fromItemId -> toItemId`); connectivity is undirected, "downstream" follows the arrows.
 */

export interface DirectedEdge {
  from: string
  to: string
}

/** Normalize connectors to plain directed edges (drops any that reference a missing endpoint). */
export function toEdges(
  connectors: { fromItemId: string; toItemId: string }[],
  present?: (id: string) => boolean,
): DirectedEdge[] {
  const ok = (id: string): boolean => (present ? present(id) : true)
  return connectors
    .filter((c) => ok(c.fromItemId) && ok(c.toItemId))
    .map((c) => ({ from: c.fromItemId, to: c.toItemId }))
}

/** Every node reachable from `seeds` treating edges as undirected - i.e. the full connected graph(s)
 * the seeds belong to. Seeds are always included (a lone node is its own one-node graph). */
export function expandToGraphs(seeds: Iterable<string>, edges: DirectedEdge[]): Set<string> {
  const undirected = new Map<string, Set<string>>()
  const link = (a: string, b: string): void => {
    if (!undirected.has(a)) undirected.set(a, new Set())
    undirected.get(a)!.add(b)
  }
  for (const e of edges) {
    link(e.from, e.to)
    link(e.to, e.from)
  }
  const out = new Set<string>()
  const stack = [...seeds]
  while (stack.length) {
    const id = stack.pop()!
    if (out.has(id)) continue
    out.add(id)
    for (const n of undirected.get(id) ?? []) if (!out.has(n)) stack.push(n)
  }
  return out
}

/**
 * The run target(s) of a graph: runnable nodes with **no other runnable node downstream**. Running a
 * target serializes its whole upstream closure, so this is the node that "produces" the graph. A
 * chain (frame → refine) targets the last; a branch to two outputs yields two targets.
 */
export function runTargets(
  graph: Set<string>,
  edges: DirectedEdge[],
  isRunnable: (id: string) => boolean,
): string[] {
  const downstream = new Map<string, Set<string>>()
  for (const e of edges) {
    if (!graph.has(e.from) || !graph.has(e.to)) continue
    if (!downstream.has(e.from)) downstream.set(e.from, new Set())
    downstream.get(e.from)!.add(e.to)
  }
  const runnable = [...graph].filter(isRunnable)
  const hasRunnableDownstream = (start: string): boolean => {
    const seen = new Set<string>([start])
    const stack = [...(downstream.get(start) ?? [])]
    while (stack.length) {
      const id = stack.pop()!
      if (seen.has(id)) continue
      seen.add(id)
      if (isRunnable(id)) return true
      for (const n of downstream.get(id) ?? []) stack.push(n)
    }
    return false
  }
  return runnable.filter((id) => !hasRunnableDownstream(id))
}
