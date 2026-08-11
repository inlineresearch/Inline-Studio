/**
 * Which of a Core node's picked files are not installed.
 *
 * A graph imported from JSON (or a shared PNG) carries the model, LoRA and VAE names the author
 * used, and those files are very often absent on the machine it lands on. The node would look fine
 * and then fail at run time, so the card marks itself instead.
 *
 * Pure, so it unit-tests without React or a backend.
 */
import type { NodeDescriptor } from '@shared/coreNodes'

export interface MissingInput {
  /** The param key, e.g. `file` or `vae`. */
  key: string
  label: string
  /** What the graph asked for but the catalog does not have. */
  value: string
}

/**
 * The picks this node makes that are not in the installed catalog.
 *
 * Empty when the answer is not knowable, so a node is never marked broken on missing information:
 * no descriptor yet (the registry is still loading), a param with no catalog behind it, or an
 * empty pick (Core auto-resolves those).
 */
export function missingInputs(
  descriptor: NodeDescriptor | undefined,
  params: Record<string, unknown> | undefined,
): MissingInput[] {
  if (!descriptor || !params) return []
  const out: MissingInput[] = []
  for (const field of descriptor.params) {
    if (!field.optionsFrom) continue
    const options = field.options
    // No options at all means the catalog has not been read yet, not that the file is absent.
    if (!options || options.length === 0) continue
    const value = params[field.key]
    if (typeof value !== 'string' || value.trim() === '') continue
    if (options.some((o) => o.value === value)) continue
    out.push({ key: field.key, label: field.label, value })
  }
  return out
}

/** One line for a tooltip, naming exactly what to install. */
export function missingInputsMessage(missing: MissingInput[]): string {
  if (missing.length === 0) return ''
  const names = missing.map((m) => `${m.label}: ${m.value}`).join(', ')
  return `Not installed, so this node cannot run yet. ${names}`
}
