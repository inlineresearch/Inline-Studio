/**
 * Copy, export, and duplicate the graph behind a node.
 *
 * The JSON is the **recipe shape**, the same one embedded in generated PNGs, so an exported file
 * can be dropped straight back onto the canvas and rebuilt by `lib/recipeGraph.buildGraphFromRecipe`
 * with no importer of its own. Entirely client-side: the board is already in the store.
 */
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import type { Recipe, RecipeConnector, RecipeItem } from '../../lib/pngRecipe'
import { copyText } from '../../lib/clipboard'
import { useMoodboardStore } from '../../store/moodboardStore'
import { expandToGraphs, toEdges } from './graphSelection'

/** Node types `buildGraphFromRecipe` can recreate. Anything else exports but will not re-import.
 * `asset` and a rendered `frame` come back as empty Load Assets nodes: the wiring rebuilds, the
 * media does not, because asset ids mean nothing in another project. */
const REBUILDABLE = new Set(['core', 'prompt', 'controlSpace', 'loader', 'frame', 'asset'])

/** Item fields the recipe keeps, per type. Mirrors `studio/recipe.py::_clean_data`. */
function cleanData(item: MoodboardItem): Record<string, unknown> {
  const data = (item.data ?? {}) as Record<string, unknown>
  switch (item.type) {
    case 'core': {
      // A recipe says how to make the image, so the node's take history is deliberately dropped.
      const core = (data.core ?? {}) as { type?: string; params?: Record<string, unknown> }
      return { core: { type: core.type, params: core.params ?? {} } }
    }
    case 'prompt':
      return { promptText: data.promptText ?? '' }
    case 'controlSpace':
      return { controlAssetId: data.controlAssetId, controlScene: data.controlScene }
    case 'loader':
      return { assetIds: data.assetIds ?? [] }
    case 'text':
    case 'layer':
    case 'director':
    case 'trim':
      return data
    default:
      return {}
  }
}

export interface GraphSlice {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
}

/** The connected graph `itemId` belongs to, items and the connectors between them. */
export function graphSlice(itemId: string): GraphSlice {
  const { items, connectors } = useMoodboardStore.getState()
  const present = new Set(items.map((i) => i.id))
  const edges = toEdges(connectors, (id) => present.has(id))
  const ids = expandToGraphs([itemId], edges)
  return {
    items: items.filter((i) => ids.has(i.id)),
    connectors: connectors.filter((c) => ids.has(c.fromItemId) && ids.has(c.toItemId)),
  }
}

/** Node types in this graph that the importer cannot rebuild, for the menu's warning. */
export function unsupportedTypes(itemId: string): string[] {
  const { items } = graphSlice(itemId)
  return [...new Set(items.map((i) => i.type).filter((t) => !REBUILDABLE.has(t)))]
}

export function graphRecipe(itemId: string): Recipe {
  const { items, connectors } = graphSlice(itemId)
  const target = items.find((i) => i.id === itemId)
  const core = (
    (target?.data ?? {}) as { core?: { type?: string; params?: Record<string, unknown> } }
  ).core
  const promptItem = items.find((i) => i.type === 'prompt')
  const recipeItems: RecipeItem[] = items.map((i) => ({
    id: i.id,
    type: i.type,
    data: cleanData(i),
    x: i.x,
    y: i.y,
    width: i.width,
    height: i.height,
    assetId: i.assetId,
    frameId: i.frameId,
  }))
  const recipeConnectors: RecipeConnector[] = connectors.map((c) => ({
    fromItemId: c.fromItemId,
    toItemId: c.toItemId,
    data: (c.data ?? {}) as Record<string, unknown>,
  }))
  return {
    version: 1,
    app: 'inline-studio',
    target: itemId,
    coreType: core?.type,
    params: core?.params ?? {},
    prompt: String(((promptItem?.data ?? {}) as { promptText?: unknown }).promptText ?? ''),
    graph: { items: recipeItems, connectors: recipeConnectors },
  }
}

export function graphJson(itemId: string): string {
  return JSON.stringify(graphRecipe(itemId), null, 2)
}

export async function copyGraphJson(itemId: string): Promise<boolean> {
  return copyText(graphJson(itemId))
}

export function exportGraphJson(itemId: string, name = 'graph'): void {
  const blob = new Blob([graphJson(itemId)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${name}.inline-graph.json`
  link.click()
  URL.revokeObjectURL(url)
}

/**
 * Copy the graph beside itself, wiring included. `duplicateItems` only recreates items, so the
 * connectors are rebuilt here through the old-id to new-id map or the copy comes back unwired.
 */
export async function duplicateGraph(itemId: string, offset = 48): Promise<number> {
  const { items, connectors } = graphSlice(itemId)
  if (items.length === 0) return 0
  const store = useMoodboardStore.getState()
  const created = await store.duplicateItems(items, { x: offset, y: offset })
  if (created.length === 0) return 0

  const idMap = useMoodboardStore.getState().lastDuplicateIdMap
  for (const connector of connectors) {
    const from = idMap.get(connector.fromItemId)
    const to = idMap.get(connector.toItemId)
    if (!from || !to) continue
    const data = (connector.data ?? {}) as { sourceHandle?: unknown; targetHandle?: unknown }
    // No undo snapshot per wire: duplicateItems already took one, and duplicating is one action.
    await store.connect(
      from,
      to,
      typeof data.sourceHandle === 'string' ? data.sourceHandle : null,
      typeof data.targetHandle === 'string' ? data.targetHandle : null,
      false,
    )
  }
  return created.length
}
