/**
 * Whether a wire between two canvas handles is allowed, checked before it is made.
 *
 * Two rules, in order. Between low-level Core ports, Core's own type rule decides (model, latent,
 * conditioning, …). Then the text rule: a Prompt node emits text and every other node emits media,
 * so a Prompt may only feed a text input and a media output may only feed a non-text one.
 */
import { portsSatisfy, type NodeDescriptor, type PortKind } from '@shared/coreNodes'
import type { MoodboardItem } from '@shared/types'

export interface WireEnds {
  source: string | null
  target: string | null
  sourceHandle?: string | null
  targetHandle?: string | null
}

/** A handle's port kind, or null when the node has no descriptor to read one from. */
export function portKindAt(
  item: MoodboardItem | undefined,
  handle: string | null | undefined,
  side: 'input' | 'output',
  descriptors: NodeDescriptor[],
): PortKind | null {
  // Control Space emits a control map, not a plain image: kinded 'control' so it can only feed a
  // gen node's Control input, never the img2img Image input (which would ignore the pose).
  if (item?.type === 'controlSpace' && side === 'output') return 'control'
  const core = item?.type === 'core' ? item.data.core : undefined
  if (!core) return null
  const descriptor = descriptors.find((d) => d.type === core.type)
  const ports = side === 'output' ? descriptor?.outputs : descriptor?.inputs
  return ports?.find((p) => p.id === handle)?.kind ?? null
}

export function canWire(
  wire: WireEnds,
  items: MoodboardItem[],
  descriptors: NodeDescriptor[],
): boolean {
  if (!wire.source || !wire.target || wire.source === wire.target) return false
  const source = items.find((it) => it.id === wire.source)
  const target = items.find((it) => it.id === wire.target)
  const srcKind = portKindAt(source, wire.sourceHandle, 'output', descriptors)
  const tgtKind = portKindAt(target, wire.targetHandle, 'input', descriptors)
  if (srcKind && tgtKind && !portsSatisfy(srcKind, tgtKind)) return false

  const sourceIsText = source?.type === 'prompt'
  // By kind, not by handle name: a Core node's text input can be called anything (Encode
  // Character's is `description`), and a fal node has no descriptor to read a kind from.
  const targetIsText = tgtKind === 'text' || (wire.targetHandle ?? undefined) === 'prompt'
  return sourceIsText === targetIsText
}
