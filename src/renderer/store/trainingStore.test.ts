import { beforeEach, describe, expect, it } from 'vitest'
import type { InlineStudioApi } from '@shared/ipc'
import { ok } from '@shared/result'
import type { TrainingHyperparams, TrainingRun } from '@shared/types'
import { setStudioClient } from '../lib/studio'
import { useTrainingStore } from './trainingStore'

const HP: TrainingHyperparams = {
  baseMode: 'turbo_adapter',
  rank: 16,
  alpha: 16,
  learningRate: 1e-4,
  steps: 100,
  batchSize: 1,
  resolution: 1024,
  saveEvery: 50,
  gpuIds: [],
}

function run(id: string): TrainingRun {
  return {
    id,
    projectId: 'p',
    datasetId: 'd',
    name: id,
    status: 'training',
    hyperparams: HP,
    outputLoraPath: null,
    progressFraction: 0,
    progressStatus: null,
    step: 0,
    totalSteps: 100,
    error: null,
    createdAt: 0,
    updatedAt: 0,
  }
}

let listRunsCalls = 0

beforeEach(() => {
  // applyDone/applyError fire off a reload, so the reducers need the backend seam injected.
  listRunsCalls = 0
  setStudioClient({
    training: {
      listRuns: async () => {
        listRunsCalls += 1
        return ok([])
      },
    },
  } as unknown as InlineStudioApi)
  useTrainingStore.setState({
    progressByRun: {},
    lossByRun: {},
    samplesByRun: {},
    systemStats: null,
    runs: [],
  })
})

describe('trainingStore reducers', () => {
  it('records progress and accumulates the loss curve', () => {
    const s = useTrainingStore.getState()
    s.applyProgress({ runId: 'r', fraction: 0.1, step: 10, totalSteps: 100, loss: 0.5 })
    s.applyProgress({ runId: 'r', fraction: 0.2, step: 20, totalSteps: 100, loss: 0.4 })
    const st = useTrainingStore.getState()
    expect(st.progressByRun['r']).toEqual({
      fraction: 0.2,
      step: 20,
      totalSteps: 100,
      status: undefined,
    })
    expect(st.lossByRun['r']).toEqual([0.5, 0.4])
  })

  it('does not push a loss point when the event has none', () => {
    useTrainingStore
      .getState()
      .applyProgress({ runId: 'r', fraction: 0.3, step: 30, totalSteps: 100 })
    expect(useTrainingStore.getState().lossByRun['r']).toBeUndefined()
  })

  it('appends sample previews per run', () => {
    const s = useTrainingStore.getState()
    s.applySample({ runId: 'r', step: 50, path: 'training_runs/r/samples/1.png' })
    s.applySample({ runId: 'r', step: 100, path: 'training_runs/r/samples/2.png' })
    expect(useTrainingStore.getState().samplesByRun['r']).toEqual([
      'training_runs/r/samples/1.png',
      'training_runs/r/samples/2.png',
    ])
  })

  it('stores host/GPU telemetry', () => {
    const stats = { cpu: 12, ramUsed: 1, ramTotal: 2, gpus: [] }
    useTrainingStore.getState().applyStats(stats)
    expect(useTrainingStore.getState().systemStats).toEqual(stats)
  })

  it('patches a run error in place and reloads so the run status lands', async () => {
    useTrainingStore.setState({ runs: [run('r')] })
    useTrainingStore.getState().applyError({ runId: 'r', error: 'boom' })
    expect(useTrainingStore.getState().runs[0]!.error).toBe('boom')
    await Promise.resolve()
    expect(listRunsCalls).toBe(1)
  })

  it('reloads runs when one finishes', async () => {
    useTrainingStore.setState({ runs: [run('r')] })
    useTrainingStore.getState().applyDone({ runId: 'r', outputLoraPath: 'loras/r.safetensors' })
    const done = useTrainingStore.getState().runs[0]!
    expect(done.status).toBe('done')
    expect(done.outputLoraPath).toBe('loras/r.safetensors')
    await Promise.resolve()
    expect(listRunsCalls).toBe(1)
  })
})
