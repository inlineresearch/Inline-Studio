import { describe, expect, it } from 'vitest'
import { IpcChannels } from './ipc'

describe('training wire contract', () => {
  it('declares every training RPC channel', () => {
    expect(IpcChannels.training.start).toBe('training:start')
    expect(Object.keys(IpcChannels.training)).toEqual(
      expect.arrayContaining([
        'listDatasets',
        'createDataset',
        'updateDataset',
        'listItems',
        'addItems',
        'removeItem',
        'setCaption',
        'autoCaption',
        'listRuns',
        'start',
        'resume',
        'cancel',
        'status',
      ]),
    )
  })

  it('declares the training + telemetry event channels', () => {
    expect(IpcChannels.events.trainingProgress).toBe('events:trainingProgress')
    expect(IpcChannels.events.trainingSample).toBe('events:trainingSample')
    expect(IpcChannels.events.trainingDone).toBe('events:trainingDone')
    expect(IpcChannels.events.trainingError).toBe('events:trainingError')
    expect(IpcChannels.events.systemStats).toBe('events:systemStats')
  })
})
