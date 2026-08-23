/**
 * Build a starter graph on the Studio canvas: a prompt node wired into a model node, either an
 * Inline Core node or a hosted fal one.
 *
 * Mirrors `recipeGraph.ts` deliberately, including `recordHistory: false` on the param writes.
 * Note that `addCoreNode`, `addPrompt` and `connect` each record their own undo entry, so a starter
 * graph unwinds in three steps rather than one; that matches how recipe rebuilds already behave.
 */
import { useGenerationStore } from '../store/generationStore'
import { useMoodboardStore } from '../store/moodboardStore'
import { useUiStore } from '../store/uiStore'
import {
  PROMPT_SOURCE_HANDLE,
  PROMPT_TARGET_HANDLE,
  starterLayout,
  type Point,
  type StarterRecipe,
} from './starterRecipes'

/**
 * Returns the created ids as `[promptId, genId]`, or `[]` if the board rejected a node (Core down).
 * A partial graph is worse than none, so a failure aborts before wiring and nothing is left dangling.
 */
export async function buildStarterGraph(recipe: StarterRecipe, centre: Point): Promise<string[]> {
  if (!recipe.coreType && !recipe.falModelId) return []
  const store = useMoodboardStore.getState()
  const at = starterLayout(centre)

  const gen = recipe.falModelId
    ? await store.addGenNode(recipe.falModelId, at.gen.x, at.gen.y)
    : await store.addCoreNode(recipe.coreType as string, at.gen.x, at.gen.y)
  if (!gen) {
    useGenerationStore.getState().setError('Could not add the model node. Is Inline Core running?')
    return []
  }
  if (recipe.coreType) {
    await store.updateItem(
      gen.id,
      { data: { ...gen.data, core: { type: recipe.coreType, params: { ...recipe.params } } } },
      false,
    )
  } else if (gen.frameId) {
    // A fal node's params live on its frame, not in the item's data blob.
    await useGenerationStore.getState().setParams(gen.frameId, { ...recipe.params })
  }

  const prompt = await store.addPrompt(at.prompt.x, at.prompt.y)
  if (!prompt) {
    // The model node stays: it is usable on its own, and deleting it would also wipe the undo entry.
    useGenerationStore.getState().setError('Could not add the prompt node.')
    return []
  }
  await store.updateItem(
    prompt.id,
    { data: { ...prompt.data, promptText: recipe.promptText } },
    false,
  )

  await store.connect(prompt.id, gen.id, PROMPT_SOURCE_HANDLE, PROMPT_TARGET_HANDLE)
  return [prompt.id, gen.id]
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

/** The training chain, pre-wired: Load Dataset -> Train LoRA -> Graph. Returns [] if any add failed. */
export async function buildTrainingStarter(centre: Point): Promise<string[]> {
  const store = useMoodboardStore.getState()
  const y = clearRow(centre, centre.x - 620, centre.x + 580, 340)
  const dataset = await store.addTrainingNode('train/dataset', centre.x - 620, y)
  const trainer = dataset && (await store.addTrainingNode('train/lora', centre.x - 240, y))
  const graph = trainer && (await store.addTrainingNode('train/loss', centre.x + 260, y))
  if (!dataset || !trainer || !graph) {
    useGenerationStore
      .getState()
      .setError('Could not add the training nodes. Is Inline Core running?')
    return []
  }
  await store.connect(dataset.id, trainer.id, 'out', 'dataset')
  await store.connect(trainer.id, graph.id, 'out', 'run')
  useUiStore.getState().revealAt(centre.x - 20, y + 170)
  return [dataset.id, trainer.id, graph.id]
}
