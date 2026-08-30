/**
 * Rebuild a canvas subgraph from a recipe (embedded in a generated image). Recreates the recipe's
 * nodes with fresh ids, offset so the producing node lands at the drop point, then re-wires the
 * links. Used by "Load graph" when an output image is dropped or a shared image is imported.
 *
 * Structure, prompt and settings transfer; the input media does not, because an asset id means
 * nothing in another project. Media-bearing sources (asset / rendered frame) and loaders holding
 * assets this project lacks land as **empty** Load Assets nodes, still wired, so the user drops
 * their own file onto one and the graph is whole.
 */
import type { MoodboardItem } from '@shared/types'
import { useMoodboardStore } from '../store/moodboardStore'
import { useGenerationStore } from '../store/generationStore'
import { useAssetStore } from '../store/assetStore'
import { paramValues, type Recipe } from './pngRecipe'

interface Point {
  x: number
  y: number
}

/** Rebuild the recipe onto the board near `drop`. Returns the number of nodes created. */
export async function buildGraphFromRecipe(recipe: Recipe, drop: Point): Promise<number> {
  const graph = recipe.graph
  if (!graph?.items?.length) return 0
  const store = useMoodboardStore.getState()

  const target = graph.items.find((i) => i.id === recipe.target) ?? graph.items[0]
  const offX = drop.x - target.x
  const offY = drop.y - target.y
  const idMap = new Map<string, string>()
  // The Library may never have been opened on this board, and an empty store would read as "this
  // project has no assets" and strip loaders that were in fact valid.
  if (!useAssetStore.getState().assets.length) await useAssetStore.getState().load()
  const known = new Set(useAssetStore.getState().assets.map((a) => a.id))
  const substituted = new Set<string>()

  for (const it of graph.items) {
    const x = it.x + offX
    const y = it.y + offY
    const data = it.data ?? {}
    let created: MoodboardItem | null = null

    if (it.type === 'core') {
      const core = data.core as { type?: string; params?: unknown } | undefined
      if (!core?.type) continue
      created = await store.addCoreNode(core.type, x, y)
      if (created) {
        // Rebuilt as plain values whichever shape the file used; the board stores values, and the
        // types are the descriptor's to say.
        const params = paramValues(core.params)
        await store.updateItem(
          created.id,
          { data: { ...created.data, core: { type: core.type, params } } },
          false,
        )
      }
    } else if (it.type === 'prompt') {
      created = await store.addPrompt(x, y)
      if (created) {
        await store.updateItem(
          created.id,
          { data: { ...created.data, promptText: String(data.promptText ?? '') } },
          false,
        )
      }
    } else if (it.type === 'controlSpace') {
      created = await store.addControlSpace(x, y)
      if (created && (data.controlAssetId || data.controlScene)) {
        await store.updateItem(created.id, { data: { ...created.data, ...data } }, false)
      }
    } else if (it.type === 'loader') {
      created = await store.addLoader(x, y)
      // Asset ids are project-scoped, so a recipe from another project names media this one has
      // never seen. Keeping them would leave a loader that looks full and resolves to nothing.
      const assetIds = ((data.assetIds as string[] | undefined) ?? []).filter((id) => known.has(id))
      if (created && assetIds.length) {
        await store.updateItem(created.id, { data: { ...created.data, assetIds } }, false)
        await store.fitLoaderToAsset(created.id, assetIds[0])
      }
    } else if (it.type === 'frame' && (data.fal as { modelId?: string } | undefined)?.modelId) {
      // A fal gen node - Core authored it with the model + params.
      const fal = data.fal as { modelId: string; params?: Record<string, unknown> }
      created = await store.addGenNode(fal.modelId, x, y)
      if (created?.frameId && fal.params) {
        await useGenerationStore.getState().setParams(created.frameId, fal.params)
      }
    } else if (
      it.type === 'train/dataset' ||
      it.type === 'train/caption' ||
      it.type === 'train/lora' ||
      it.type === 'train/loss'
    ) {
      created = await store.addTrainingNode(it.type, x, y)
      // Settings travel; the dataset and run they were bound to do not.
      if (created && Object.keys(data).length) {
        await store.updateItem(created.id, { data: { ...created.data, ...data } }, false)
      }
    } else if (it.type === 'resource') {
      created = await store.addResource(x, y)
    } else if (it.type === 'asset' || it.type === 'frame') {
      // Media does not travel between projects, but the wiring should. An empty Load Assets node
      // stands in, so the user drops their own clip onto it and the graph is whole again.
      created = await store.addLoader(x, y)
      if (created) substituted.add(it.id)
    } else {
      continue // text/etc. - nothing to rebuild in v1
    }

    if (created) idMap.set(it.id, created.id)
  }

  for (const c of graph.connectors ?? []) {
    const from = idMap.get(c.fromItemId)
    const to = idMap.get(c.toItemId)
    if (!from || !to) continue
    const cd = c.data as { sourceHandle?: string; targetHandle?: string }
    // A loader standing in for an asset carries the loader's own output handle, not the one the
    // original node named; null lets the board fall back to it.
    const src = substituted.has(c.fromItemId) ? null : (cd.sourceHandle ?? null)
    await store.connect(from, to, src, cd.targetHandle ?? null)
  }

  return idMap.size
}
