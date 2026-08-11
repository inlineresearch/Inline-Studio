import { beforeEach, describe, expect, it } from 'vitest'
import type { InlineStudioApi } from '@shared/ipc'
import { ok } from '@shared/result'
import type { ActivityRun } from '@shared/types'
import { setCoreConnected } from '../lib/connection'
import { setStudioClient } from '../lib/studio'
import { subscribeActivityEvents, useActivityStore } from './activityStore'
import { useGenerationStore } from './generationStore'

function run(runId: string, status: ActivityRun['status'] = 'running'): ActivityRun {
  return {
    runId,
    kind: 'generation',
    engine: 'core',
    origin: 'studio',
    status,
    title: 'Z-Image',
    fraction: 0.5,
    statusLabel: null,
    queuePosition: null,
    queuedAt: 1,
    startedAt: 2,
    endedAt: null,
    error: null,
    takeId: null,
    projectId: 'p1',
    projectName: 'Alpha',
    projectPath: '/tmp/Alpha',
    itemId: 'i1',
    surface: 'studio',
  }
}

let historyCalls = 0
let cancelled: string[] = []

beforeEach(() => {
  historyCalls = 0
  cancelled = []
  setStudioClient({
    activity: {
      list: async () => ok([]),
      history: async () => {
        historyCalls += 1
        return ok([])
      },
      cancel: async (runId: string) => {
        cancelled.push(runId)
        return ok(undefined)
      },
      clearHistory: async () => ok(undefined),
    },
    generation: { active: async () => ok([]) },
    events: { onActivityChanged: () => () => undefined },
  } as unknown as InlineStudioApi)
  useActivityStore.setState({ live: [], history: [], error: null })
})

describe('activityStore', () => {
  it('replaces the live list from a snapshot', () => {
    useActivityStore.getState().applySnapshot([run('r1'), run('r2', 'queued')])
    expect(useActivityStore.getState().live.map((r) => r.runId)).toEqual(['r1', 'r2'])

    useActivityStore.getState().applySnapshot([run('r2', 'queued')])
    expect(useActivityStore.getState().live.map((r) => r.runId)).toEqual(['r2'])
  })

  it('refreshes history when a run leaves the live list', async () => {
    useActivityStore.getState().applySnapshot([run('r1'), run('r2')])
    expect(historyCalls).toBe(0)

    useActivityStore.getState().applySnapshot([run('r2')])
    await Promise.resolve()
    expect(historyCalls).toBe(1)
  })

  it('does not refetch history while the same runs keep progressing', async () => {
    useActivityStore.getState().applySnapshot([run('r1')])
    useActivityStore.getState().applySnapshot([run('r1')])
    await Promise.resolve()
    expect(historyCalls).toBe(0)
  })

  it('drops a cancelled run locally before the round trip lands', async () => {
    useActivityStore.getState().applySnapshot([run('r1'), run('r2')])
    await useActivityStore.getState().cancel('r1')
    expect(useActivityStore.getState().live.map((r) => r.runId)).toEqual(['r2'])
    expect(cancelled).toEqual(['r1'])
  })

  it('cancelAll cancels every live run by id', () => {
    // One call per run, because training and fal runs do not share the generation cancel path.
    useActivityStore.getState().applySnapshot([run('r1'), run('r2', 'queued')])
    return useActivityStore
      .getState()
      .cancelAll()
      .then(() => {
        expect(cancelled).toEqual(['r1', 'r2'])
        expect(useActivityStore.getState().live).toEqual([])
      })
  })

  it('cancelAll is a no-op with nothing running', async () => {
    await useActivityStore.getState().cancelAll()
    expect(cancelled).toEqual([])
  })

  it('re-asks Core for the truth when the socket reconnects', async () => {
    // Events sent while the socket was down are gone, so a run that finished during the outage
    // would otherwise sit in the list as running forever.
    setCoreConnected(false)
    const stop = subscribeActivityEvents()
    await Promise.resolve()
    const before = historyCalls

    setCoreConnected(true)
    await Promise.resolve()

    expect(historyCalls).toBeGreaterThan(before)
    stop()
  })

  it('does not resync when the socket drops', async () => {
    setCoreConnected(true)
    const stop = subscribeActivityEvents()
    await Promise.resolve()
    const before = historyCalls

    setCoreConnected(false)
    await Promise.resolve()

    expect(historyCalls).toBe(before)
    stop()
  })
})

describe('generationStore run bookkeeping', () => {
  beforeEach(() => {
    useGenerationStore.setState({
      busyByFrame: { a: true, b: true },
      progressByFrame: { a: 0.4, b: 0.7 },
      statusByFrame: { a: 'sampling', b: 'loading' },
    })
  })

  it('finishing one run leaves the others alone', () => {
    // The old finishAll() cleared every node, which breaks as soon as a queue is visible.
    useGenerationStore.getState().finishRun('a')
    const state = useGenerationStore.getState()
    expect(state.busyByFrame['a']).toBe(false)
    expect(state.progressByFrame['a']).toBeNull()
    expect(state.busyByFrame['b']).toBe(true)
    expect(state.progressByFrame['b']).toBe(0.7)
  })

  it('hydrateActive replaces local state, so a Core restart clears every node', async () => {
    // Core came back with nothing running. A merge left these two spinning forever.
    await useGenerationStore.getState().hydrateActive()
    expect(useGenerationStore.getState().busyByFrame).toEqual({})
    expect(useGenerationStore.getState().statusByFrame).toEqual({})
  })

  it('reset clears every node, for a project close', () => {
    useGenerationStore.getState().reset()
    expect(useGenerationStore.getState().busyByFrame).toEqual({})
    expect(useGenerationStore.getState().progressByFrame).toEqual({})
  })
})
