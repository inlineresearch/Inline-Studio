import type { CorePendingRun, CoreTakeRef } from '@shared/types'

/** Which slot the strip is showing. `current` is the node's live settings, never a take. */
export type SlotId = 'current' | string

/** The node data these read. Narrowed so the helpers stay testable without a MoodboardItem. */
export interface SlotSource {
  params?: Record<string, unknown>
  output?: CoreTakeRef
  outputs?: CoreTakeRef[]
  pending?: CorePendingRun
}

/** One entry in the take strip. Takes are newest first, behind Current when a run is unlanded. */
export interface Slot {
  id: SlotId
  /** Absent on Current, whose run produced no media: edited, in flight, cancelled or failed. */
  take?: CoreTakeRef
  state: 'take' | 'running' | 'cancelled' | 'failed' | 'draft'
}

/** Renders newest first, behind one Current slot while a run has not landed. See docs. */
export function buildSlots(core: SlotSource, busy: boolean, edited = false): Slot[] {
  const takes = core.outputs ?? (core.output ? [core.output] : [])
  const history = takes.map((t) => ({ id: t.takeId, take: t, state: 'take' as const }))
  const state = slotState(activePending(core)?.status, busy, edited)
  return state ? [{ id: 'current', state }, ...history] : history
}

/** What the one non-take slot reads as, most recent fact first. */
function slotState(
  status: CorePendingRun['status'] | undefined,
  busy: boolean,
  edited: boolean,
): Exclude<Slot['state'], 'take'> | null {
  if (busy || status === 'running') return 'running'
  if (edited || status === 'draft') return 'draft'
  if (status === 'cancelled' || status === 'failed') return status
  return null
}

/** The snapshot still describing something unfinished. Every reader goes through it: a landed run
 *  leaves `status: 'running'` behind for good, because clearing it destroyed takes. */
export function activePending(core: SlotSource): CorePendingRun | undefined {
  const takes = core.outputs ?? (core.output ? [core.output] : [])
  return landed(core, takes) ? undefined : core.pending
}

/** Whether the snapshotted run already produced one of these takes, making its slot obsolete. */
function landed(core: SlotSource, takes: CoreTakeRef[]): boolean {
  const started = core.pending?.startedAt
  if (started === undefined || core.pending?.status !== 'running') return false
  // `createdAt` is absent on renders made before it was tracked; those predate any snapshot.
  return takes.some((t) => t.createdAt !== undefined && t.createdAt >= started)
}

/** Whether the live recipe has moved away from the last thing rendered or submitted.
 *
 *  Both baselines are the node's own params. A take's `params` are the runner's, which cannot
 *  answer this; `nodeParams` is absent on older takes, and those say so rather than guess. */
export function hasEdits(core: SlotSource, livePrompt?: string): boolean {
  const baseline = activePending(core) ?? takeBaseline(core)
  if (!baseline) return false
  if ((baseline.prompt ?? '') !== (livePrompt ?? '')) return true
  return !sameParams(withoutSeed(core.params), withoutSeed(baseline.params))
}

function takeBaseline(
  core: SlotSource,
): { params?: Record<string, unknown>; prompt?: string } | undefined {
  const newest = (core.outputs ?? [])[0]
  return newest?.nodeParams ? { params: newest.nodeParams, prompt: newest.prompt } : undefined
}

function withoutSeed(params: Record<string, unknown> | undefined): Record<string, unknown> {
  const out = { ...(params ?? {}) }
  delete out.seed
  return out
}

function sameParams(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])]
  return keys.every((k) => JSON.stringify(a[k]) === JSON.stringify(b[k]))
}

/** The recipe a slot restores: a take's, or the in-flight run's for Current. */
export function slotRecipe(
  core: SlotSource,
  slot: SlotId,
): { params?: Record<string, unknown>; prompt?: string } | undefined {
  if (slot !== 'current') return (core.outputs ?? []).find((t) => t.takeId === slot)
  return activePending(core)
}

/** The prompt under the strip: the browsed take's, or the live one when Current is shown. */
export function slotPrompt(
  core: SlotSource,
  slot: SlotId,
  livePrompt?: string,
): string | undefined {
  if (slot === 'current') return livePrompt
  return (core.outputs ?? []).find((t) => t.takeId === slot)?.prompt
}

/** Which take's media fills the preview. Current has none of its own, so the active output stands. */
export function slotMedia(core: SlotSource, slot: SlotId): CoreTakeRef | undefined {
  if (slot === 'current') return core.output
  return (core.outputs ?? []).find((t) => t.takeId === slot) ?? core.output
}

/** Settings to push onto the node, seed withheld: pinning one turns on the node cache.
 *
 *  `restorable` drops the installed-file dropdowns, which a take records as runner ids. */
export function applyableParams(
  recipe: { params?: Record<string, unknown> },
  restorable?: ReadonlySet<string>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(recipe.params ?? {})) {
    if (key === 'seed') continue
    if (restorable && !restorable.has(key)) continue
    out[key] = value
  }
  return out
}

/** Which of a node's params a take may write back: everything but the installed-files dropdowns. */
export function restorableKeys(
  params: readonly { key: string; optionsFrom?: string }[] | undefined,
): ReadonlySet<string> {
  return new Set((params ?? []).filter((p) => !p.optionsFrom).map((p) => p.key))
}
