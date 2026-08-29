import { describe, expect, it } from 'vitest'
import type { CoreTakeRef } from '@shared/types'
import {
  applyableParams,
  buildSlots,
  hasEdits,
  slotMedia,
  slotPrompt,
  slotRecipe,
} from './takeSlots'

const take = (id: string, prompt: string): CoreTakeRef => ({
  takeId: id,
  filePath: `${id}.png`,
  kind: 'image',
  prompt,
  params: { steps: 20, seed: 7 },
})

describe('the take strip', () => {
  it('shows Current while a run has not landed, and not otherwise', () => {
    expect(buildSlots({}, false)).toEqual([])
    expect(buildSlots({}, true).map((s) => s.id)).toEqual(['current'])
    const core = { outputs: [take('a', 'x')] }
    expect(buildSlots(core, true).map((s) => s.id)).toEqual(['current', 'a'])
    expect(buildSlots(core, false).map((s) => s.id)).toEqual(['a'])
  })

  it('keeps the slot after a stop, because the snapshot is the only copy of those settings', () => {
    // Selecting a take overwrites the node's params. Compare against history mid-render, cancel,
    // and without this the settings you submitted are gone with no way back to them.
    const core = { outputs: [take('a', 'x')] }
    for (const status of ['cancelled', 'failed'] as const) {
      const slots = buildSlots(
        { ...core, pending: { startedAt: 1, status, params: { steps: 4 } } },
        false,
      )
      expect(slots[0].id).toBe('current')
      expect(slots[0].state).toBe(status)
      expect(
        slotRecipe({ ...core, pending: { startedAt: 1, status, params: { steps: 4 } } }, 'current')
          ?.params,
      ).toEqual({ steps: 4 })
    }
  })

  it('drops the slot only when the run lands, its take carrying the same recipe', () => {
    // Takes are newest first, so when the run ends its take occupies the position Current held.
    const during = buildSlots({ outputs: [take('a', 'x')] }, true)
    const after = buildSlots({ outputs: [take('b', 'new'), take('a', 'x')] }, false)
    expect(during[0].id).toBe('current')
    expect(after[0].id).toBe('b')
  })

  it('keeps Current up while the snapshot says running, before any progress event lands', () => {
    // A run inside a long model load emits nothing for minutes; without this the slot would not
    // appear until the first progress tick, which on H3 is well over a minute.
    expect(buildSlots({ pending: { startedAt: 1, status: 'running' } }, false)[0].state).toBe(
      'running',
    )
  })

  it('restores the in-flight run from Current and the take from a take', () => {
    const core = {
      outputs: [take('a', 'old')],
      pending: { startedAt: 1, status: 'running' as const, prompt: 'live' },
    }
    expect(slotRecipe(core, 'a')?.prompt).toBe('old')
    expect(slotRecipe(core, 'current')?.prompt).toBe('live')
  })

  it("shows the live prompt on Current and the take's prompt when browsing", () => {
    const core = { outputs: [take('a', 'old')] }
    expect(slotPrompt(core, 'current', 'what will run next')).toBe('what will run next')
    expect(slotPrompt(core, 'a', 'what will run next')).toBe('old')
  })

  it('never restores the seed, because pinning one silently freezes re-generation', () => {
    expect(applyableParams({ params: { steps: 20, seed: 7 } })).toEqual({ steps: 20 })
  })

  it('falls back to the active output when a slot has gone', () => {
    const core = { output: take('a', 'x'), outputs: [take('a', 'x')] }
    expect(slotMedia(core, 'deleted')?.takeId).toBe('a')
    expect(slotMedia(core, 'current')?.takeId).toBe('a')
  })

  it('raises the slot as soon as settings move away from the last render', () => {
    // Completed renders append; one slot covers everything else, so an edit and the render it
    // becomes are the same entry rather than two.
    const core = { params: { steps: 20 }, outputs: [take('a', 'old')] }
    expect(hasEdits(core, 'old')).toBe(false)
    expect(hasEdits({ ...core, params: { steps: 30 } }, 'old')).toBe(true)
    expect(hasEdits(core, 'a new prompt')).toBe(true)
    expect(buildSlots(core, false, true)[0].state).toBe('draft')
  })

  it('does not call a restored take an edit, because restore withholds the seed', () => {
    // Live params keep the node's own seed after a restore. Counting it would leave every node
    // permanently flagged as edited the moment you browsed history.
    const core = { params: { steps: 20, seed: 999 }, outputs: [take('a', 'old')] }
    expect(hasEdits(core, 'old')).toBe(false)
  })

  it('has no draft state before the first render, with nothing to differ from', () => {
    expect(hasEdits({ params: { steps: 20 } }, 'anything')).toBe(false)
  })

  it('drops a running slot once a newer take exists, without anyone writing to clear it', () => {
    // The completion event used to clear the snapshot, which wrote the node back from the client's
    // copy and dropped the take Core had just appended: a finished render vanished, file and all.
    const pending = { startedAt: 1000, status: 'running' as const }
    const landed = { ...take('b', 'new'), createdAt: 1500 }
    expect(buildSlots({ pending, outputs: [landed] }, false).map((s) => s.id)).toEqual(['b'])
    const older = { ...take('a', 'old'), createdAt: 500 }
    expect(buildSlots({ pending, outputs: [older] }, false)[0].id).toBe('current')
  })

  it('keeps a stopped slot even once newer takes exist', () => {
    // Only a *running* snapshot is superseded by a take. A cancelled one is the sole copy of those
    // settings and has to survive until the next run replaces it.
    const pending = { startedAt: 1000, status: 'cancelled' as const }
    const landed = { ...take('b', 'new'), createdAt: 1500 }
    expect(buildSlots({ pending, outputs: [landed] }, false)[0].state).toBe('cancelled')
  })
})
