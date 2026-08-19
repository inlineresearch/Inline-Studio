/**
 * Copy, export, and duplicate the graph behind a node.
 *
 * The JSON is the **recipe shape**, the same one embedded in generated PNGs, so an exported file
 * can be dropped straight back onto the canvas and rebuilt by `lib/recipeGraph.buildGraphFromRecipe`
 * with no importer of its own. Entirely client-side: the board is already in the store.
 */
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import { RECIPE_VERSION } from '../../lib/pngRecipe'
import type {
  Recipe,
  RecipeConnector,
  RecipeItem,
  RecipeModel,
  RecipeParam,
  RecipeParamType,
} from '../../lib/pngRecipe'
import { downloadUrl } from '@shared/modelRefs'
import { useModelRequirementsStore } from '../../store/modelRequirementsStore'
import { useModelRegistryStore } from '../../store/modelRegistryStore'
import { copyText } from '../../lib/clipboard'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import { expandToGraphs, toEdges } from './graphSelection'

/** Node types `buildGraphFromRecipe` can recreate. Anything else exports but will not re-import.
 * `asset` and a rendered `frame` come back as empty Load Assets nodes: the wiring rebuilds, the
 * media does not, because asset ids mean nothing in another project. */
const REBUILDABLE = new Set([
  'core',
  'prompt',
  'controlSpace',
  'loader',
  'frame',
  'asset',
  'train/dataset',
  'train/caption',
  'train/lora',
  'train/loss',
  'resource',
])

/**
 * The params a run would actually use, not only the ones the user touched.
 *
 * A select left alone stores nothing: the node face shows the file the engine resolved, and the
 * run uses it, but the item's params stay empty. Exported as-is, a graph naming no weight anywhere
 * cannot be rebuilt on another machine and reports needing no models at all. Core serves each
 * field's `default` already resolved to the file it would load, so that is what gets recorded.
 */
function effectiveParams(core: {
  type?: string
  params?: Record<string, unknown>
}): Record<string, unknown> {
  const stored = { ...(core.params ?? {}) }
  const type = String(core.type ?? '')
  const descriptor = useCoreNodesStore.getState().descriptors.find((d) => d.type === type)
  if (!descriptor) return stored
  for (const field of descriptor.params) {
    const value = stored[field.key]
    if (value !== undefined && value !== null && value !== '') continue
    const resolved = field.default !== '' ? field.default : field.options?.[0]?.value
    if (resolved !== undefined && resolved !== null && resolved !== '') {
      stored[field.key] = resolved
    }
  }
  return stored
}

/** Item fields the recipe keeps, per type. Mirrors `studio/recipe.py::_clean_data`. */
function cleanData(item: MoodboardItem): Record<string, unknown> {
  const data = (item.data ?? {}) as Record<string, unknown>
  switch (item.type) {
    case 'core': {
      // A recipe says how to make the image, so the node's take history is deliberately dropped.
      const core = (data.core ?? {}) as { type?: string; params?: Record<string, unknown> }
      return { core: coreEntry(core) }
    }
    case 'prompt':
      return { promptText: data.promptText ?? '' }
    // Training nodes keep their settings, never their bindings: a dataset and a run are rows in
    // this project's database, so they name nothing in the project the recipe lands in.
    case 'train/lora':
      return { hyperparams: data.hyperparams ?? {} }
    case 'train/caption':
      return { overwrite: data.overwrite ?? false, captioner: data.captioner ?? '' }
    case 'train/dataset':
    case 'train/loss':
    case 'resource':
      return {}
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

/**
 * A node's core entry: its type, each param as {type, value}, and where its models come from.
 *
 * The kind travels with the value because a bare filename forces a reader to guess the folder from
 * the param's name, and the coordinates travel because an export is otherwise only fetchable on a
 * machine whose registry still carries the same entry.
 */
function coreEntry(core: { type?: string; params?: Record<string, unknown> }): {
  type: string
  params: Record<string, RecipeParam>
  models?: RecipeModel[]
} {
  const type = String(core.type ?? '')
  const descriptor = useCoreNodesStore.getState().descriptors.find((d) => d.type === type)
  const kinds = new Map((descriptor?.params ?? []).map((p) => [p.key, p.kind ?? 'string']))
  const params = effectiveParams(core)
  const typed: Record<string, RecipeParam> = {}
  const wanted = new Map<string, string>()
  for (const [key, value] of Object.entries(params)) {
    const type = (kinds.get(key) ?? 'string') as RecipeParamType
    typed[key] = { type, value }
    if ((type === 'model' || type === 'character') && String(value)) {
      wanted.set(String(value), '')
    }
  }
  const registry = useModelRegistryStore.getState().entries
  // Which folders the params already spoke for. A node's declared requirements name its *default*
  // build, so a node set to klein-9b would otherwise export klein-4b beside it.
  const covered = new Set(
    [...wanted.keys()].map(
      (name) => registry.find((e) => e.filename.toLowerCase() === name.toLowerCase())?.category,
    ),
  )
  // What the node needs without naming it, which no param can carry.
  for (const component of useModelRequirementsStore.getState().byType[type]?.components ?? []) {
    if (component.optional || covered.has(component.category)) continue
    wanted.set(component.localPath.split('/').pop() ?? component.localPath, component.category)
  }
  const models = [...wanted].map(([name, directory]) => {
    const entry = registry.find((e) => e.filename.toLowerCase() === name.toLowerCase())
    // The node's own category wins: the registry may not carry the file, and the folder it
    // belongs in is still known.
    return {
      directory: directory || entry?.category || '',
      name,
      url: entry ? downloadUrl(entry) : '',
    }
  })
  return models.length ? { type, params: typed, models } : { type, params: typed }
}

export async function graphRecipe(itemId: string): Promise<Recipe> {
  const { items, connectors } = graphSlice(itemId)
  // Loaded rather than assumed: a node scrolled out of view was never mounted, so its
  // requirements would be absent and its models would silently not be exported.
  const types = [
    ...new Set(
      items
        .map((i) => (i.data as { core?: { type?: string } }).core?.type)
        .filter((v): v is string => !!v),
    ),
  ]
  await Promise.all([
    ...types.map((type) => useModelRequirementsStore.getState().load(type)),
    useModelRegistryStore.getState().entries.length
      ? Promise.resolve()
      : useModelRegistryStore.getState().load(),
  ])
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
    version: RECIPE_VERSION,
    app: 'inline-studio',
    target: itemId,
    coreType: core?.type,
    params: core?.params ?? {},
    prompt: String(((promptItem?.data ?? {}) as { promptText?: unknown }).promptText ?? ''),
    graph: { items: recipeItems, connectors: recipeConnectors },
  }
}

export async function graphJson(itemId: string): Promise<string> {
  return JSON.stringify(await graphRecipe(itemId), null, 2)
}

export async function copyGraphJson(itemId: string): Promise<boolean> {
  return copyText(await graphJson(itemId))
}

export async function exportGraphJson(itemId: string, name = 'graph'): Promise<void> {
  const blob = new Blob([await graphJson(itemId)], { type: 'application/json' })
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
