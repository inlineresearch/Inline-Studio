/**
 * Serialize a canvas subgraph into an Inline Core graph (schemaVersion 1). Walks the connector graph
 * upstream from a target node's closure and emits each node with its typed input edges:
 *   - a 'core' item   -> its Core node type + params (handles are already Core port ids)
 *   - a 'prompt' item -> an `input/text` source node
 *   - an 'asset' item -> an `input/image` source node (local path ref)
 * Connectors become typed edges (source output port -> target input port). The pure `serializeWorkflow`
 * is unit-tested; `buildWorkflowGraph` wraps it with the board + asset lookups.
 */
import { join } from 'node:path'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import { getDb, getOpenProjectFolder } from '../db'
import { listBoard } from '../moodboard/store'

export interface CoreGraphDoc {
  schemaVersion: number
  nodes: Array<Record<string, unknown>>
}

type Edge = { from: string; output: string }

/** The Core source-node output port for a canvas item, remapping non-core handles. */
function sourceOutputPort(
  source: MoodboardItem | undefined,
  sourceHandle: string | undefined,
): string {
  if (source?.type === 'prompt') return 'text'
  if (source?.type === 'asset') return 'image'
  // A 'core' item's handles already are the Core port ids.
  return sourceHandle ?? 'out'
}

function edgesFor(
  itemId: string,
  connectors: MoodboardConnector[],
  byId: Map<string, MoodboardItem>,
): Record<string, Edge> {
  const inputs: Record<string, Edge> = {}
  for (const c of connectors) {
    if (c.toItemId !== itemId) continue
    const targetHandle = (c.data?.targetHandle as string | undefined) ?? 'in'
    const sourceHandle = c.data?.sourceHandle as string | undefined
    inputs[targetHandle] = {
      from: c.fromItemId,
      output: sourceOutputPort(byId.get(c.fromItemId), sourceHandle),
    }
  }
  return inputs
}

function itemToNode(
  item: MoodboardItem,
  connectors: MoodboardConnector[],
  byId: Map<string, MoodboardItem>,
  resolveAssetPath: (assetId: string) => string | null,
): Record<string, unknown> | null {
  if (item.type === 'core' && item.data.core) {
    return {
      id: item.id,
      type: item.data.core.type,
      params: item.data.core.params,
      inputs: edgesFor(item.id, connectors, byId),
    }
  }
  if (item.type === 'prompt') {
    return { id: item.id, type: 'input/text', params: { text: item.data.promptText ?? '' } }
  }
  if (item.type === 'asset' && item.assetId) {
    const path = resolveAssetPath(item.assetId)
    if (!path) return null
    return { id: item.id, type: 'input/image', params: { asset: { ref: 'path', path } } }
  }
  return null
}

function upstreamClosure(target: string, connectors: MoodboardConnector[]): Set<string> {
  const seen = new Set<string>()
  const stack = [target]
  while (stack.length > 0) {
    const id = stack.pop() as string
    if (seen.has(id)) continue
    seen.add(id)
    for (const c of connectors) {
      if (c.toItemId === id && !seen.has(c.fromItemId)) stack.push(c.fromItemId)
    }
  }
  return seen
}

/** Pure: build the Core graph for `targetItemId` from the given board + asset resolver. */
export function serializeWorkflow(
  targetItemId: string,
  items: MoodboardItem[],
  connectors: MoodboardConnector[],
  resolveAssetPath: (assetId: string) => string | null,
): { graph: CoreGraphDoc; target: string } {
  const byId = new Map(items.map((i) => [i.id, i]))
  const closure = upstreamClosure(targetItemId, connectors)
  const nodes: Array<Record<string, unknown>> = []
  for (const id of closure) {
    const item = byId.get(id)
    if (!item) continue
    const node = itemToNode(item, connectors, byId, resolveAssetPath)
    if (node) nodes.push(node)
  }
  return { graph: { schemaVersion: 1, nodes }, target: targetItemId }
}

/** Build the Core graph for a canvas node from the open project's board. */
export function buildWorkflowGraph(targetItemId: string): { graph: CoreGraphDoc; target: string } {
  const { items, connectors } = listBoard()
  const folder = getOpenProjectFolder()
  return serializeWorkflow(targetItemId, items, connectors, (assetId) => {
    const row = getDb().prepare('SELECT file_path FROM assets WHERE id = ?').get(assetId) as
      | { file_path: string }
      | undefined
    return row && folder ? join(folder, row.file_path) : null
  })
}
