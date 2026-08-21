/**
 * One entry point for "does this need models we do not have?", so the four places that ask cannot
 * drift apart: a dropped workflow, a generation node, the trainer, and a character build.
 */
import { coreNodeTypes, modelFilenames } from '@shared/modelRefs'
import { useModelRegistryStore, type ModelRequest } from '../store/modelRegistryStore'

/** Check named files. Opens the popup when any is absent, and answers how many were. */
export async function checkModels(wanted: ModelRequest[], reason: string): Promise<number> {
  return useModelRegistryStore.getState().check(wanted, reason)
}

export interface DeclaredComponent {
  localPath: string
  category: string
  present: boolean
  optional?: boolean
}

const basename = (path: string): string => path.split('/').pop() ?? path

/**
 * What an imported graph still needs: the weights it names, plus a node's declared components for
 * anything it does not name.
 *
 * A category the graph already names a file for is settled. A node offers alternatives within one
 * category (FLUX.2 lists klein-4b, klein-9b and dev), so a workflow tagged klein-9b must not be
 * told klein-4b is missing just because that is the node's default.
 */
export function graphModelRequests(
  named: string[],
  componentsByType: Record<string, DeclaredComponent[]>,
): ModelRequest[] {
  const wanted: ModelRequest[] = named.map((filename) => ({ filename }))
  const seen = new Set(named)

  for (const components of Object.values(componentsByType)) {
    const covered = new Set(
      components.filter((c) => seen.has(basename(c.localPath))).map((c) => c.category),
    )
    for (const component of components) {
      if (component.present || component.optional || covered.has(component.category)) continue
      const filename = basename(component.localPath)
      if (seen.has(filename)) continue
      seen.add(filename)
      wanted.push({ filename, category: component.category })
    }
  }
  return wanted
}

/** Check everything an exported graph needs. See `graphModelRequests` for the rule. */
export async function checkGraphModels(graph: unknown, reason: string): Promise<number> {
  const types = coreNodeTypes(graph)
  const { useModelRequirementsStore } = await import('../store/modelRequirementsStore')
  await Promise.all(types.map((type) => useModelRequirementsStore.getState().load(type)))
  const byType = useModelRequirementsStore.getState().byType
  const components = Object.fromEntries(types.map((type) => [type, byType[type]?.components ?? []]))
  return checkModels(graphModelRequests(modelFilenames(graph), components), reason)
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
