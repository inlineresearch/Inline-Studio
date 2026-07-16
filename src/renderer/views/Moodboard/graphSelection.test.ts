import { describe, expect, it } from 'vitest'
import { expandToGraphs, runTargets, toEdges } from './graphSelection'

// load(m) ─┐
// load(v) ─┼─▶ zimage ──▶ preview
// prompt ──┘
const CONNECTORS = [
  { fromItemId: 'm', toItemId: 'z' },
  { fromItemId: 'v', toItemId: 'z' },
  { fromItemId: 'p', toItemId: 'z' },
  { fromItemId: 'z', toItemId: 'prev' },
]

describe('expandToGraphs', () => {
  it('selecting any node expands to the whole connected graph', () => {
    const edges = toEdges(CONNECTORS)
    const graph = expandToGraphs(['m'], edges)
    expect(graph).toEqual(new Set(['m', 'v', 'p', 'z', 'prev']))
    // From any other member the graph is identical.
    expect(expandToGraphs(['prev'], edges)).toEqual(graph)
  })

  it('a lone node is its own one-node graph', () => {
    expect(expandToGraphs(['solo'], toEdges(CONNECTORS))).toEqual(new Set(['solo']))
  })

  it('unions two disjoint graphs from a multi-node seed', () => {
    const edges = toEdges([...CONNECTORS, { fromItemId: 'a', toItemId: 'b' }])
    expect(expandToGraphs(['m', 'a'], edges)).toEqual(
      new Set(['m', 'v', 'p', 'z', 'prev', 'a', 'b']),
    )
  })
})

describe('runTargets', () => {
  const edges = toEdges(CONNECTORS)
  // Only zimage is a runnable generation node; loaders/prompt/preview are not.
  const isRunnable = (id: string): boolean => id === 'z'

  it('is the runnable node with nothing runnable downstream (the preview does not count)', () => {
    const graph = expandToGraphs(['m'], edges)
    expect(runTargets(graph, edges, isRunnable)).toEqual(['z'])
  })

  it('a chain of two generation nodes targets the last', () => {
    const chain = toEdges([
      { fromItemId: 'f1', toItemId: 'f2' },
      { fromItemId: 'f2', toItemId: 'out' },
    ])
    const graph = expandToGraphs(['f1'], chain)
    const runnable = (id: string): boolean => id === 'f1' || id === 'f2'
    expect(runTargets(graph, chain, runnable)).toEqual(['f2'])
  })

  it('no runnable node yields no target', () => {
    const graph = expandToGraphs(['m'], edges)
    expect(runTargets(graph, edges, () => false)).toEqual([])
  })
})
