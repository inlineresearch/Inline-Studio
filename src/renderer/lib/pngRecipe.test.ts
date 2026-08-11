import { describe, expect, it } from 'vitest'
import { parseRecipeBytes, parseRecipeJson, type Recipe } from './pngRecipe'

const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10]

/** Build a minimal PNG: signature + one tEXt chunk (keyword\0text) + IEND. Enough to exercise the
 * parser (CRC is not validated by the reader). */
function pngWithText(keyword: string, text: string): Uint8Array {
  const enc = (s: string): number[] => Array.from(s, (c) => c.charCodeAt(0))
  const chunk = (type: string, data: number[]): number[] => {
    const len = data.length
    return [
      (len >>> 24) & 255,
      (len >>> 16) & 255,
      (len >>> 8) & 255,
      len & 255,
      ...enc(type),
      ...data,
      0,
      0,
      0,
      0, // zeroed CRC
    ]
  }
  const textData = [...enc(keyword), 0, ...enc(text)]
  return new Uint8Array([...PNG_SIG, ...chunk('tEXt', textData), ...chunk('IEND', [])])
}

describe('parseRecipeBytes', () => {
  const recipe: Recipe = {
    version: 1,
    app: 'inline-studio',
    target: 'z1',
    prompt: 'a fox',
    graph: { items: [{ id: 'z1', type: 'core', data: {}, x: 0, y: 0 }], connectors: [] },
  }

  it('extracts and parses the inline-studio recipe from a tEXt chunk', () => {
    const png = pngWithText('inline-studio', JSON.stringify(recipe))
    expect(parseRecipeBytes(png)).toEqual(recipe)
  })

  it('returns null for a PNG with no recipe chunk', () => {
    expect(parseRecipeBytes(pngWithText('Comment', 'made elsewhere'))).toBeNull()
  })

  it('returns null when the chunk is not our app (foreign JSON)', () => {
    const png = pngWithText('inline-studio', JSON.stringify({ app: 'other', prompt: 'x' }))
    expect(parseRecipeBytes(png)).toBeNull()
  })

  it('returns null for non-PNG bytes and malformed JSON', () => {
    expect(parseRecipeBytes(new Uint8Array([1, 2, 3]))).toBeNull()
    expect(parseRecipeBytes(pngWithText('inline-studio', '{not json'))).toBeNull()
  })
})

describe('parseRecipeJson', () => {
  const graph = { items: [{ id: 'a', type: 'core', data: {}, x: 0, y: 0 }], connectors: [] }

  it('reads an exported graph file', () => {
    const text = JSON.stringify({ version: 1, app: 'inline-studio', target: 'a', graph })
    expect(parseRecipeJson(text)?.target).toBe('a')
  })

  it('rejects JSON from some other app', () => {
    // The `app` guard is what stops an unrelated .json drop from being rebuilt as a graph.
    expect(parseRecipeJson(JSON.stringify({ app: 'something-else', graph }))).toBeNull()
  })

  it('rejects malformed JSON rather than throwing', () => {
    expect(parseRecipeJson('{not json')).toBeNull()
  })

  it('rejects a bare array', () => {
    expect(parseRecipeJson('[1,2,3]')).toBeNull()
  })
})
