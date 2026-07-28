/**
 * Rebuild a canvas subgraph from a recipe (embedded in a generated image). Recreates the recipe's
 * nodes with fresh ids, offset so the producing node lands at the drop point, then re-wires the
 * links. Used by "Load graph" when an output image is dropped or a shared image is imported.
 *
 * v1 rebuilds Core-generation graphs (core / prompt / control-space / loader nodes). Media-bearing
 * nodes (asset/frame) and cross-project media refs are skipped or left dangling - the structure,
 * prompt and settings transfer; the actual input images don't (a follow-up embeds those bytes).
 */
import type { MoodboardItem } from '@shared/types'
import { useMoodboardStore } from '../store/moodboardStore'
import { useGenerationStore } from '../store/generationStore'
import type { Recipe } from './pngRecipe'

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

  for (const it of graph.items) {
    const x = it.x + offX
    const y = it.y + offY
    const data = it.data ?? {}
    let created: MoodboardItem | null = null

    if (it.type === 'core') {
      const core = data.core as { type?: string; params?: Record<string, unknown> } | undefined
      if (!core?.type) continue
      created = await store.addCoreNode(core.type, x, y)
      if (created) {
        await store.updateItem(
          created.id,
          { data: { ...created.data, core: { type: core.type, params: core.params ?? {} } } },
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
      const assetIds = (data.assetIds as string[] | undefined) ?? []
      if (created && assetIds.length) {
        await store.updateItem(created.id, { data: { ...created.data, assetIds } }, false)
      }
    } else if (it.type === 'frame') {
      // A fal gen node (Core authored it with the model + params); rendered-frame image sources
      // have no `fal` block and are skipped (their media doesn't transfer across a rebuild).
      const fal = data.fal as { modelId?: string; params?: Record<string, unknown> } | undefined
      if (!fal?.modelId) continue
      created = await store.addGenNode(fal.modelId, x, y)
      if (created?.frameId && fal.params) {
        await useGenerationStore.getState().setParams(created.frameId, fal.params)
      }
    } else {
      continue // asset/text/etc. need existing project media - skipped in v1
    }

    if (created) idMap.set(it.id, created.id)
  }

  for (const c of graph.connectors ?? []) {
    const from = idMap.get(c.fromItemId)
    const to = idMap.get(c.toItemId)
    if (!from || !to) continue
    const cd = c.data as { sourceHandle?: string; targetHandle?: string }
    await store.connect(from, to, cd.sourceHandle ?? null, cd.targetHandle ?? null)
  }

  return idMap.size
}
