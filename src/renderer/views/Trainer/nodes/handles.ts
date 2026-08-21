/** Port ids for the training nodes. `dataset` flows Load Dataset → Caption → Train LoRA, whose two
 * outputs are the adapter and its step/loss series. One colour per port kind, as everywhere else. */
export const DATASET_HANDLE = 'dataset'
/** The step/loss series the curve plots; named `run` in graphs made before it had its own kind. */
export const METRICS_HANDLE = 'metrics'
/** The trained adapter, so Attach Adapter can file it as a character's payload. */
export const LORA_HANDLE = 'lora'

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
