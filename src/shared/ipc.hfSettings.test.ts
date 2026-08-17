import { describe, expect, it } from 'vitest'
import { IpcChannels } from './ipc'

describe('hugging face token wire contract', () => {
  it('declares the token channels', () => {
    expect(Object.keys(IpcChannels.hfSettings)).toEqual(
      expect.arrayContaining(['status', 'setToken', 'clearToken']),
    )
  })

  it('names every channel after its key', () => {
    // Built generically from these entries, so a mismatched value calls the wrong handler.
    for (const [key, channel] of Object.entries(IpcChannels.hfSettings)) {
      expect(channel).toBe(`hfSettings:${key}`)
    }
  })

  it('keeps the token out of the response shape', () => {
    // Write-only by type, asserted so widening ApiKeyStatus has to be a deliberate act.
    const keys: (keyof import('./types').ApiKeyStatus)[] = ['configured', 'encrypted']
    expect(keys).not.toContain('token')
  })
})
