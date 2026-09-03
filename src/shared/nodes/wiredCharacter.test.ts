import { describe, expect, it } from 'vitest'
import type { MoodboardConnector, MoodboardItem } from '../types'
import { wiredCharacterFile } from './wiredCharacter'

describe('wiredCharacterFile', () => {
  const CONNECT = (from: string): MoodboardConnector[] =>
    [
      { toItemId: 'gen', fromItemId: from, data: { targetHandle: 'character' } },
    ] as unknown as MoodboardConnector[]

  function source(id: string, type: string, params: Record<string, unknown>): MoodboardItem {
    return { id, type: 'core', data: { core: { type, params } } } as unknown as MoodboardItem
  }

  it('reads Load Character and Write .char, which name the file differently', () => {
    // Mirrors Core: `character/load` keeps it under `file`, `character/write` under `filename`.
    expect(
      wiredCharacterFile('gen', CONNECT('a'), [source('a', 'character/load', { file: 'e.char' })]),
    ).toBe('e.char')
    expect(
      wiredCharacterFile('gen', CONNECT('b'), [source('b', 'character/write', { filename: 'e' })]),
    ).toBe('e.char')
  })

  it('reduces a typed path the way the library does', () => {
    expect(
      wiredCharacterFile('gen', CONNECT('c'), [
        source('c', 'character/write', { filename: '  /core/models/emmy-s500-v6' }),
      ]),
    ).toBe('emmy-s500-v6.char')
  })

  it('names nothing for a character that has never been written', () => {
    // Encode emits an identity that only exists inside a running graph - it has no file to count.
    expect(
      wiredCharacterFile('gen', CONNECT('d'), [source('d', 'character/encode', { name: 'e' })]),
    ).toBeNull()
    expect(wiredCharacterFile('gen', [], [])).toBeNull()
  })
})
