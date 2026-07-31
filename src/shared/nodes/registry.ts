/**
 * The registry of available generation nodes. Adding a fal model = create a `NodeDef` file and
 * append it here; the rest of the app (canvas palette, executor, param UI) is data-driven off this.
 */
import type { NodeDef } from './types'
import { GPT_IMAGE_2 } from './gptImage2'
import { NANO_BANANA_2 } from './nanoBanana2'
import { NANO_BANANA_PRO } from './nanoBananaPro'
import { LTX_I2V } from './ltxImageToVideo'
import { SEEDANCE_T2V } from './seedanceT2V'
import { SEEDANCE_I2V } from './seedanceI2V'
import { SEEDANCE_REF2V } from './seedanceRef2V'
import { KREA_V2 } from './kreaV2'
import { FLUX_2, FLUX_2_EDIT } from './flux2'
import { SONILO_V2M } from './soniloVideoToMusic'
import { SONILO_T2M } from './soniloTextToMusic'
import { SONILO_V2V } from './soniloVideoToVideo'
import { MINIMAX_H3_T2V } from './minimaxH3T2V'
import { MINIMAX_H3_I2V } from './minimaxH3I2V'
import { MINIMAX_H3_REF2V } from './minimaxH3Ref2V'

export const NODE_DEFS: readonly NodeDef[] = [
  GPT_IMAGE_2,
  NANO_BANANA_2,
  NANO_BANANA_PRO,
  LTX_I2V,
  SEEDANCE_T2V,
  SEEDANCE_I2V,
  SEEDANCE_REF2V,
  MINIMAX_H3_T2V,
  MINIMAX_H3_I2V,
  MINIMAX_H3_REF2V,
  KREA_V2,
  FLUX_2,
  FLUX_2_EDIT,
  SONILO_V2M,
  SONILO_T2M,
  SONILO_V2V,
]

/** Look up a node def by its model id (`Frame.modelId`); undefined if unknown. */
export function getNodeDef(id: string): NodeDef | undefined {
  return NODE_DEFS.find((def) => def.id === id)
}

/** All registered node defs, in palette order. */
export function listNodeDefs(): readonly NodeDef[] {
  return NODE_DEFS
}

// ── Provider grouping ──────────────────────────────────────────────────────────
// A model's "provider" is its owner - the first segment of the fal id (openai/…, fal-ai/…,
// bytedance/…, krea/…). Used to group models under headers in the picker and Add menu.

const OWNER_LABELS: Record<string, string> = {
  'fal-ai': 'fal',
  openai: 'OpenAI',
  bytedance: 'ByteDance',
  krea: 'Krea',
  sonilo: 'Sonilo',
  // H3's ids have no `fal-ai/` prefix, unlike MiniMax's older Hailuo endpoints.
  minimax: 'MiniMax',
}

export function modelOwner(id: string): string {
  return id.split('/')[0] ?? 'other'
}

export function modelOwnerLabel(id: string): string {
  const owner = modelOwner(id)
  return OWNER_LABELS[owner] ?? owner
}

export interface ModelGroup {
  owner: string
  label: string
  defs: NodeDef[]
}

/** Group defs by owner, preserving first-seen (registry) order within and across groups. */
export function groupByOwner(defs: readonly NodeDef[]): ModelGroup[] {
  const groups: ModelGroup[] = []
  for (const def of defs) {
    const owner = modelOwner(def.id)
    let group = groups.find((g) => g.owner === owner)
    if (!group) {
      group = { owner, label: modelOwnerLabel(def.id), defs: [] }
      groups.push(group)
    }
    group.defs.push(def)
  }
  return groups
}
