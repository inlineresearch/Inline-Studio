/**
 * Build a starter graph on the Studio canvas: a prompt node wired into a model node.
 *
 * Mirrors `recipeGraph.ts` deliberately, including `recordHistory: false` on the param writes.
 * Note that `addCoreNode`, `addPrompt` and `connect` each record their own undo entry, so a starter
 * graph unwinds in three steps rather than one; that matches how recipe rebuilds already behave.
 */
import { useGenerationStore } from '../store/generationStore'
import { useMoodboardStore } from '../store/moodboardStore'
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
  if (!recipe.coreType) return []
  const store = useMoodboardStore.getState()
  const at = starterLayout(centre)

  const gen = await store.addCoreNode(recipe.coreType, at.gen.x, at.gen.y)
  if (!gen) {
    useGenerationStore.getState().setError('Could not add the model node. Is Inline Core running?')
    return []
  }
  await store.updateItem(
    gen.id,
    { data: { ...gen.data, core: { type: recipe.coreType, params: { ...recipe.params } } } },
    false,
  )

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
