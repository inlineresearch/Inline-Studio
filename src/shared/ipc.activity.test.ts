import { describe, expect, it } from 'vitest'
import { IpcChannels } from './ipc'

describe('activity wire contract', () => {
  it('declares every activity RPC channel', () => {
    expect(Object.keys(IpcChannels.activity)).toEqual(
      expect.arrayContaining(['list', 'history', 'cancel', 'clearHistory']),
    )
    expect(IpcChannels.activity.list).toBe('activity:list')
    expect(IpcChannels.activity.cancel).toBe('activity:cancel')
  })

  it('declares the models tree channel for the Models panel', () => {
    expect(IpcChannels.models.tree).toBe('models:tree')
  })

  it('declares the activity + cancellation event channels', () => {
    // The web client derives `studio().events.onX` from these keys, so the names are the contract.
    expect(IpcChannels.events.activityChanged).toBe('events:activityChanged')
    expect(IpcChannels.events.generationCancelled).toBe('events:generationCancelled')
  })

  it('keeps every channel value namespaced by its group', () => {
    for (const [key, channel] of Object.entries(IpcChannels.activity)) {
      expect(channel).toBe(`activity:${key}`)
    }
  })
})
