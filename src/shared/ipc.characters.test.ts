import { describe, expect, it } from 'vitest'
import { IpcChannels } from './ipc'

describe('characters wire contract', () => {
  it('declares every character RPC channel', () => {
    expect(Object.keys(IpcChannels.characters)).toEqual(
      expect.arrayContaining([
        'list',
        'create',
        'get',
        'rename',
        'setDescription',
        'addRefs',
        'removeRef',
        'delete',
        'createFromTake',
      ]),
    )
  })

  it('keeps every channel value namespaced by its group', () => {
    // The web client builds `studio().characters.<key>` generically from these entries, so a
    // mismatched value silently produces a method that calls the wrong handler.
    for (const [key, channel] of Object.entries(IpcChannels.characters)) {
      expect(channel).toBe(`characters:${key}`)
    }
  })

  it('declares the build channel', () => {
    expect(IpcChannels.characters.build).toBe('characters:build')
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
