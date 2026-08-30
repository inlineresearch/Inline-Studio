import { describe, expect, it } from 'vitest'
import type { CoreTakeRef } from '@shared/types'
import {
  activePending,
  applyableParams,
  buildSlots,
  hasEdits,
  restorableKeys,
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

  it('keeps showing the running slot when edits are made mid-render', () => {
    // Progress is the live fact while the GPU is busy; the edit waits its turn.
    const core = {
      params: { steps: 9 },
      pending: { startedAt: 1, status: 'running' as const, params: { steps: 4 } },
    }
    expect(buildSlots(core, false, true)[0].state).toBe('running')
  })

  it("measures edits against the node's own params, never the runner's", () => {
    // A take's `params` are what the runner received: different keys, and `model` is a runner id
    // rather than the checkpoint filename. Measured against those, untouched nodes read as edited
    // and real edits to keys the runner does not take read as clean - which is what let a click
    // destroy them. `nodeParams` is the node's own, recorded beside them.
    const rendered: CoreTakeRef = {
      takeId: 'a',
      filePath: 'a.mp4',
      kind: 'video',
      prompt: 'a lake',
      params: { model: 'minimax-h3-ref2va', width: 768, seed: 11 },
      nodeParams: { model: 'h3_fp8.safetensors', width: 768, vae: 'v.safetensors', seed: -1 },
    }
    const core = { params: { ...rendered.nodeParams }, outputs: [rendered] }
    expect(hasEdits(core, 'a lake')).toBe(false)
    // An edit to a key the runner never receives still counts.
    expect(
      hasEdits({ ...core, params: { ...core.params, vae: 'other.safetensors' } }, 'a lake'),
    ).toBe(true)
    expect(hasEdits(core, 'a different prompt')).toBe(true)
  })

  it('cannot answer for a take rendered before node params were recorded', () => {
    // Saying "not edited" is the honest answer; guessing from the runner's params is what broke.
    const core = { params: { width: 512 }, outputs: [take('a', 'old')] }
    expect(hasEdits(core, 'anything')).toBe(false)
  })

  it('never writes an installed-file dropdown back from a take', () => {
    // A take's `model` is the runner id, not the checkpoint filename the node picked, so merging it
    // in left the node reporting its diffusion model missing the moment a previous take was clicked.
    const restorable = restorableKeys([
      { key: 'width' },
      { key: 'steps' },
      { key: 'model', optionsFrom: 'diffusion_models' },
      { key: 'vae', optionsFrom: 'vae' },
    ])
    const applied = applyableParams(
      { params: { model: 'minimax-h3-ref2va', vae: 'x', width: 768, steps: 50, seed: 3 } },
      restorable,
    )
    expect(applied).toEqual({ width: 768, steps: 50 })
  })

  it('stops treating a run as running once its take has landed', () => {
    // The snapshot is never cleared, so a finished run leaves `status: running` in the data for
    // good. Masking that in the strip alone was not enough: the draft capture read the raw status
    // and refused to snapshot for the rest of the node's life, so an edit on a node that had run
    // once was still destroyed by a click. Every reader goes through `activePending` now.
    const pending = { startedAt: 1000, status: 'running' as const, params: { steps: 4 } }
    const landedTake = { ...take('b', 'new'), createdAt: 1500 }
    expect(activePending({ pending, outputs: [landedTake] })).toBeUndefined()

    // Still running while nothing newer exists.
    const older = { ...take('a', 'old'), createdAt: 500 }
    expect(activePending({ pending, outputs: [older] })?.status).toBe('running')
  })

  it('falls back to the landed take as the baseline once the run is over', () => {
    const rendered: CoreTakeRef = {
      ...take('b', 'new'),
      createdAt: 1500,
      nodeParams: { steps: 8 },
    }
    const core = {
      params: { steps: 8 },
      outputs: [rendered],
      pending: { startedAt: 1000, status: 'running' as const, params: { steps: 4 } },
    }
    expect(hasEdits(core, 'new')).toBe(false)
    expect(hasEdits({ ...core, params: { steps: 9 } }, 'new')).toBe(true)
  })
})
