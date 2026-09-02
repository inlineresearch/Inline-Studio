/**
 * The pre-wired character chains the Characters panel drops onto the canvas.
 *
 * Mirrors `recipeGraph.ts` deliberately, including `recordHistory: false` on the param writes.
 * Note that `addCoreNode` and `connect` each record their own undo entry, so a chain unwinds in
 * several steps rather than one; that matches how recipe rebuilds already behave.
 */
import { useGenerationStore } from '../store/generationStore'
import { useMoodboardStore } from '../store/moodboardStore'
import { useUiStore } from '../store/uiStore'

export interface Point {
  x: number
  y: number
}

/**
 * A y where a chain spanning `left`..`right` lands on nothing. Starter chains are dropped at the
 * viewport centre, which on a canvas that already has work on it is squarely on top of it.
 */
function clearRow(centre: Point, left: number, right: number, height: number): number {
  const items = useMoodboardStore.getState().items
  let y = centre.y
  // One pass per collision: each push clears the item it hit, and the list is finite.
  for (let i = 0; i < items.length; i += 1) {
    const hit = items.find(
      (it) => it.x < right && it.x + it.width > left && it.y < y + height && it.y + it.height > y,
    )
    if (!hit) break
    y = hit.y + hit.height + 60
  }
  return y
}

/**
 * The character chain, pre-wired: Load Assets -> Encode -> Verify References -> Compile -> Write.
 * Wired for a reference model; a Krea 2 style character swaps the payload node for Train LoRA +
 * Attach Adapter, which is why the payload step is its own node rather than folded into Encode.
 */
export async function buildCharacterStarter(
  centre: Point,
  assetIds: string[] = [],
): Promise<string[]> {
  const store = useMoodboardStore.getState()
  const y = clearRow(centre, centre.x - 780, centre.x + 780, 340)
  const images = await store.addLoader(centre.x - 780, y)
  if (images && assetIds.length > 0) await store.addLoaderAssets(images.id, assetIds)
  const encode = images && (await store.addCoreNode('character/encode', centre.x - 470, y))
  const verify = encode && (await store.addCoreNode('character/verify-refs', centre.x - 160, y))
  const refs = verify && (await store.addCoreNode('character/references', centre.x + 150, y))
  const write = refs && (await store.addCoreNode('character/write', centre.x + 460, y))
  if (!images || !encode || !verify || !refs || !write) {
    useGenerationStore
      .getState()
      .setError('Could not add the character nodes. Is Inline Core running?')
    return []
  }
  await store.connect(images.id, encode.id, 'image', 'images')
  await store.connect(encode.id, verify.id, 'character', 'character')
  await store.connect(verify.id, refs.id, 'character', 'character')
  // Write takes the verified character, never the raw one: a payload node compiles from the doc
  // Write hands it, so wiring Encode straight here would save the set nothing checked.
  await store.connect(verify.id, write.id, 'character', 'character')
  await store.connect(refs.id, write.id, 'payload', 'payloads')
  useUiStore.getState().revealAt(centre.x, y + 170)
  return [images.id, encode.id, verify.id, refs.id, write.id]
}

/** Load -> Edit -> Verify References -> Write .char, for changing one that is already saved. */
export async function buildCharacterEditChain(file: string, centre: Point): Promise<string[]> {
  const store = useMoodboardStore.getState()
  const y = clearRow(centre, centre.x - 620, centre.x + 620, 340)
  const load = await store.addCoreNode('character/load', centre.x - 620, y)
  const edit = load && (await store.addCoreNode('character/edit', centre.x - 310, y))
  const verify = edit && (await store.addCoreNode('character/verify-refs', centre.x, y))
  const write = verify && (await store.addCoreNode('character/write', centre.x + 310, y))
  if (!load || !edit || !verify || !write) {
    useGenerationStore
      .getState()
      .setError('Could not add the character nodes. Is Inline Core running?')
    return []
  }
  await store.updateItem(
    load.id,
    { data: { ...load.data, core: { type: 'character/load', params: { file } } } },
    false,
  )
  await store.connect(load.id, edit.id, 'character', 'character')
  await store.connect(edit.id, verify.id, 'character', 'character')
  await store.connect(verify.id, write.id, 'character', 'character')
  useUiStore.getState().revealAt(centre.x - 180, y + 170)
  return [load.id, edit.id, verify.id, write.id]
}
