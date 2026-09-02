import { describe, expect, it } from 'vitest'
import { IpcChannels } from './ipc'

describe('workflow catalogue wire contract', () => {
  it('declares the catalogue channels', () => {
    expect(Object.keys(IpcChannels.workflows)).toEqual(
      expect.arrayContaining(['list', 'detail', 'event', 'markPrompted']),
    )
  })

  it('names every channel after its key', () => {
    // Built generically from these entries, so a mismatched value calls the wrong handler.
    for (const [key, channel] of Object.entries(IpcChannels.workflows)) {
      expect(channel).toBe(`workflows:${key}`)
    }
  })
})
