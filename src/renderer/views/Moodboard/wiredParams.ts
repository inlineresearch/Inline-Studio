/**
 * Which of a node's params are being overridden by a wire, and by what.
 *
 * Several params are also input ports: a gen node's `model`/`vae`/`text_encoder` accept a loader,
 * and Encode Character's `description` accepts a Prompt. Wired, the input wins at run time and the
 * typed value is ignored - so a panel that shows only what was typed promises a value the run will
 * not use.
 */
import type { NodeDescriptor } from '@shared/coreNodes'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'

export interface WiredParam {
  /** What is driving it, for the note under the field ("Prompt", "Load Diffusion Model"). */
  from: string
  /** The value the run will use, when the canvas can resolve it. */
  text: string
  /**
   * True when the wire supplies nothing and the typed value still applies. Text inputs fall back
   * (`wired or typed`); a component handle does not, so an empty one still overrides.
   */
  fallsBack: boolean
}

/** Params of `itemId` that a wire is driving, keyed by param key. */
export function wiredParams(
  itemId: string,
  descriptor: NodeDescriptor | undefined,
  items: MoodboardItem[],
  connectors: MoodboardConnector[],
): Map<string, WiredParam> {
  const out = new Map<string, WiredParam>()
  if (!descriptor) return out
  const paramKeys = new Set(descriptor.params.map((p) => p.key))
  const portKind = new Map(descriptor.inputs.map((p) => [p.id, p.kind]))

  for (const connector of connectors) {
    if (connector.toItemId !== itemId) continue
    const handle = (connector.data?.targetHandle as string | undefined) ?? ''
    if (!handle || !paramKeys.has(handle) || !portKind.has(handle)) continue
    const source = items.find((it) => it.id === connector.fromItemId)
    if (!source) continue
    const text = sourceValue(source)
    out.set(handle, {
      from: sourceLabel(source),
      text,
      fallsBack: portKind.get(handle) === 'text' && text.trim() === '',
    })
  }
  return out
}

/** What the upstream node contributes, when the canvas can know it without running the graph. */
function sourceValue(source: MoodboardItem): string {
  if (source.type === 'prompt') return String(source.data.promptText ?? '')
  if (source.type === 'core') return String(source.data.core?.params?.file ?? '')
  return ''
}

function sourceLabel(source: MoodboardItem): string {
  if (source.type === 'prompt') return 'Prompt'
  if (source.type === 'core') return String(source.data.core?.type ?? 'the wired node')
  return 'the wired node'
}
