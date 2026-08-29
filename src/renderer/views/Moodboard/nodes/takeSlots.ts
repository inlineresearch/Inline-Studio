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

/** Every render this node produced, newest first, behind Current while a run has not landed.
 *
 * Current holds the settings its run was submitted with, which live nowhere else. It survives a
 * cancel and a failure deliberately: selecting a take overwrites the node's params, so after
 * comparing against history the snapshot is the only copy of what you had, and dropping it on stop
 * threw away exactly the thing the slot exists to protect. A landed run needs no slot, because its
 * take carries the same recipe. */
export function buildSlots(core: SlotSource, busy: boolean, edited = false): Slot[] {
  const takes = core.outputs ?? (core.output ? [core.output] : [])
  const history = takes.map((t) => ({ id: t.takeId, take: t, state: 'take' as const }))
  // Read from the takes rather than cleared by the completion event. Clearing meant writing the
  // node back from the client's copy, which at that moment predated the take Core had just
  // appended - so the write dropped it, and the render was lost with only its file left on disk.
  const pending = landed(core, takes) ? undefined : core.pending
  const status = pending?.status
  const state: Slot['state'] | null =
    busy || status === 'running'
      ? 'running'
      : status === 'cancelled' || status === 'failed'
        ? status
        : // Settings changed since the last render and not yet run. The same slot becomes the
          // running one on Generate, so an edit and its render are one entry, not two.
          edited || status === 'draft'
          ? 'draft'
          : null
  return state ? [{ id: 'current', state }, ...history] : history
}

/** Whether the snapshotted run already produced one of these takes, making its slot obsolete. */
function landed(core: SlotSource, takes: CoreTakeRef[]): boolean {
  const started = core.pending?.startedAt
  if (started === undefined || core.pending?.status !== 'running') return false
  // `createdAt` is absent on renders made before it was tracked; those predate any snapshot.
  return takes.some((t) => t.createdAt !== undefined && t.createdAt >= started)
}

/** Whether the node's live recipe has moved away from its newest take.
 *
 * Compared without the seed, which `applyableParams` withholds on restore: counting it would make
 * every node read as edited the moment it was restored from history. */
export function hasEdits(core: SlotSource, livePrompt?: string): boolean {
  const newest = (core.outputs ?? [])[0]
  // Nothing to have moved away from yet, so a node's first run is not an "edit".
  if (!newest) return false
  if ((newest.prompt ?? '') !== (livePrompt ?? '')) return true
  return !sameParams(applyableParams({ params: core.params }), applyableParams(newest))
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
  return core.pending
}

/** The prompt to show under the strip: the browsed take's, or the live one when Current is shown.
 *
 * The line used to always render the active take's prompt, so typing a new prompt without
 * generating left the node face advertising the old one. */
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

/** Settings to push onto the node, seed withheld.
 *
 * A pinned seed makes every re-generation identical and turns on the node cache, so connection and
 * control changes stop taking effect until it is reset. Reusing one is a separate deliberate act. */
export function applyableParams(recipe: {
  params?: Record<string, unknown>
}): Record<string, unknown> {
  const out = { ...(recipe.params ?? {}) }
  delete out.seed
  return out
}
