import { describe, expect, it } from 'vitest'
import { IpcChannels } from './ipc'

describe('characters wire contract', () => {
  it('keeps every channel value namespaced by its group', () => {
    // The web client builds `studio().characters.<key>` generically from these entries, so a
    // mismatched value silently produces a method that calls the wrong handler.
    for (const [key, channel] of Object.entries(IpcChannels.characters)) {
      expect(channel).toBe(`characters:${key}`)
    }
  })

  it('offers browsing and applying, never a second way to edit', () => {
    // Two paths to edit a character would drift; the canvas chain is the one that exists.
    // `applyFal` reads one out for an endpoint - a fal node builds its request in the browser, so
    // it has to ask for the compiled references rather than reaching into the file itself.
    expect(Object.keys(IpcChannels.characters).sort()).toEqual([
      'applyFal',
      'createFromTake',
      'delete',
      'list',
      'sweepResult',
    ])
  })

  it('declares the library-changed event channel', () => {
    // `events.onCharactersChanged` is derived from this key, so the name is the contract.
    expect(IpcChannels.events.charactersChanged).toBe('events:charactersChanged')
  })

  it('declares the encode-progress event channel', () => {
    // Built at runtime, so a rename would not fail typecheck; it would just stop delivering.
    expect(IpcChannels.events.characterProgress).toBe('events:characterProgress')
  })
})
