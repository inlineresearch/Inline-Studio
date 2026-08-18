/**
 * Which weight files a graph asks for. Exported graphs carry them as plain param values, so a
 * dropped workflow can be checked against the registry without loading any descriptors.
 */

const WEIGHT_SUFFIXES = ['.safetensors', '.sft', '.gguf', '.ckpt', '.pt', '.pth', '.onnx']

export function looksLikeWeightFile(value: unknown): value is string {
  if (typeof value !== 'string' || !value) return false
  const name = value.toLowerCase()
  return WEIGHT_SUFFIXES.some((suffix) => name.endsWith(suffix))
}

/** Every weight filename referenced anywhere in a value, deduped and in first-seen order. */
export function modelFilenames(value: unknown, found = new Set<string>()): string[] {
  if (looksLikeWeightFile(value)) {
    // A param may carry a path; the registry matches on the name it lands under.
    found.add(value.split(/[\\/]/).pop() as string)
  } else if (Array.isArray(value)) {
    for (const item of value) modelFilenames(item, found)
  } else if (value && typeof value === 'object') {
    for (const item of Object.values(value)) modelFilenames(item, found)
  }
  return [...found]
}

/**
 * The Core node types an exported graph uses, deduped.
 *
 * A graph's params only name a weight when the user picked one. Left on "auto" - which is how a
 * loader is exported when the engine resolves it - the graph names no file at all, so the filename
 * sweep finds nothing and a workflow that needs several GB reports needing none. The node types are
 * what carry the declared requirements, so they are collected too.
 */
export function coreNodeTypes(value: unknown, found = new Set<string>()): string[] {
  if (Array.isArray(value)) {
    for (const item of value) coreNodeTypes(item, found)
  } else if (value && typeof value === 'object') {
    const core = (value as { core?: { type?: unknown } }).core
    if (core && typeof core.type === 'string' && core.type) found.add(core.type)
    for (const item of Object.values(value)) coreNodeTypes(item, found)
  }
  return [...found]
}
