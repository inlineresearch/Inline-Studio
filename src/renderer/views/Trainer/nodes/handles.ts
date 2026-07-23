/** Port ids for the Trainer canvas. `dataset` flows Load Dataset → Caption → Trainer; `run` flows
 * Trainer → Graph. Colours match the handle styling convention (one colour per port kind). */
export const DATASET_HANDLE = 'dataset'
export const RUN_HANDLE = 'run'

/** Walk a node's incoming `dataset` edge to the dataset it was wired to, if any. */
export function wiredDatasetId(
  itemId: string,
  connectors: { fromItemId: string; toItemId: string }[],
  items: { id: string; data: { datasetId?: string | null } }[],
): string | null {
  const seen = new Set<string>()
  let current = itemId
  // Follow the chain upstream (Trainer ← Caption ← Load Dataset) until a node names a dataset.
  while (!seen.has(current)) {
    seen.add(current)
    const incoming = connectors.find((c) => c.toItemId === current)
    if (!incoming) return null
    const source = items.find((i) => i.id === incoming.fromItemId)
    if (!source) return null
    if (source.data.datasetId) return source.data.datasetId
    current = source.id
  }
  return null
}
