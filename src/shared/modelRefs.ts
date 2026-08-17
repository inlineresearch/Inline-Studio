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
