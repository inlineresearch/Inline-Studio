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

/** A legacy full-path pick reduced to the bare name the select serves, mirroring `resolve_picked`. */
export function basename(value: string): string {
  return value.split(/[\\/]/).pop() ?? value
}

/**
 * The picks this node makes that are not in the installed catalog.
 *
 * Empty when the answer is not knowable, so a node is never marked broken on missing information:
 * no descriptor yet (the registry is still loading), a param with no catalog behind it, or an
 * empty pick (Core auto-resolves those). An empty catalog is knowable, and means absent.
 */
export function missingInputs(
  descriptor: NodeDescriptor | undefined,
  params: Record<string, unknown> | undefined,
): MissingInput[] {
  if (!descriptor || !params) return []
  const out: MissingInput[] = []
  for (const field of descriptor.params) {
    if (!field.optionsFrom) continue
    // An empty list means the category holds nothing, which is the strongest reason to flag a
    // pick rather than to stay quiet: a descriptor only reaches the client once Core has scanned,
    // so "still loading" is the `!descriptor` case above, not this one.
    const options = field.options ?? []
    const value = params[field.key]
    if (typeof value !== 'string' || value.trim() === '') continue
    if (options.some((o) => o.value === value || o.value === basename(value))) continue
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

/**
 * The catalog's options plus the pick itself when the catalog does not have it.
 *
 * A native select whose value matches no option renders blank, so a graph naming a model this
 * machine lacks lost the name it came with - and with it the one piece of information needed to go
 * and fetch the right file. Keeping it listed, and saying it is absent, is what the red border on
 * the node is already claiming.
 */
export function optionsWithPick(
  options: readonly { value: string; label: string }[],
  picked: string,
): { value: string; label: string }[] {
  const listed = [...options]
  if (!picked || listed.some((o) => o.value === picked)) return listed
  return [{ value: picked, label: `${picked} (not installed)` }, ...listed]
}
