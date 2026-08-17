/**
 * One entry point for "does this need models we do not have?", so the four places that ask cannot
 * drift apart: a dropped workflow, a generation node, the trainer, and a character build.
 */
import { modelFilenames } from '@shared/modelRefs'
import { useModelRegistryStore, type ModelRequest } from '../store/modelRegistryStore'

/** Check named files. Opens the popup when any is absent, and answers how many were. */
export async function checkModels(wanted: ModelRequest[], reason: string): Promise<number> {
  return useModelRegistryStore.getState().check(wanted, reason)
}

/** Check everything an exported graph references. Its params carry the filenames directly. */
export async function checkGraphModels(graph: unknown, reason: string): Promise<number> {
  return checkModels(
    modelFilenames(graph).map((filename) => ({ filename })),
    reason,
  )
}

/** Check a node's declared requirements, which already know the folder each file belongs in. */
export async function checkComponentModels(
  components: { localPath: string; category: string; present: boolean; optional?: boolean }[],
  reason: string,
): Promise<number> {
  return checkModels(
    components
      .filter((c) => !c.present && !c.optional)
      .map((c) => ({
        filename: c.localPath.split('/').pop() ?? c.localPath,
        category: c.category,
      })),
    reason,
  )
}
