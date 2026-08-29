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
  // Derived, never cleared by the completion event. Clearing meant writing the node back from the
  // client's copy, which at that moment predated the take Core had just appended, so the write
  // dropped it and the render was lost with only its file left on disk.
  const state = slotState(activePending(core)?.status, busy, edited)
  return state ? [{ id: 'current', state }, ...history] : history
}

/** What the one non-take slot reads as, most recent fact first.
 *
 * A render in flight outranks everything, since its progress is the live fact. An edit outranks a
 * stopped run because you made it afterwards: a cancelled run whose settings you have since changed
 * is a draft, not a cancelled run. */
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

/** The snapshot that still describes something unfinished, or nothing once its run has landed.
 *
 * Every reader must go through this. The snapshot is deliberately never cleared - clearing it meant
 * writing the node back from a stale client copy, which destroyed the take that had just landed -
 * so a finished run leaves `status: 'running'` behind in the data forever. Masking that in the strip
 * alone was not enough: the draft capture read the raw status and so refused to snapshot anything
 * for the rest of the node's life, which is how an edit could still be lost on a node that had run
 * once. */
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

/** Whether the node's live recipe has moved away from the last thing it rendered or submitted.
 *
 * Both baselines are the node's own params, which is the only sound comparison. A take's `params`
 * are the *runner's* resolved ones - different keys, and `model` holds a runner id rather than the
 * checkpoint filename - so measuring against those answered wrongly in both directions: nodes that
 * had never been touched read as edited, and real edits to keys the runner does not take read as
 * clean, which is what let a click destroy them.
 *
 * `nodeParams` is absent on takes rendered before it was recorded; those cannot answer the question
 * and say so, rather than guessing. */
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

/** Settings to push onto the node: the seed withheld, and anything the node cannot accept dropped.
 *
 * A pinned seed makes every re-generation identical and turns on the node cache, so connection and
 * control changes stop taking effect until it is reset. Reusing one is a separate deliberate act.
 *
 * `restorable` is the node's own param keys, minus the ones filled from an installed-files catalog.
 * A take records the *runner's* params, where `model` is a runner id like `minimax-h3-ref2va` and
 * not the checkpoint filename the node's dropdown holds - so merging it in blindly left the node
 * reporting its diffusion model missing. A take cannot say which file was used, so it does not. */
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
