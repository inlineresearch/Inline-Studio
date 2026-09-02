import { describe, expect, it } from 'vitest'
import type { MoodboardItem } from '@shared/types'
import { boundsOf, IMPORT_GUTTER, placeImport, recipeTarget } from './graphPlacement'

const item = (x: number, y: number, width = 200, height = 120): MoodboardItem =>
  ({ id: `${x}:${y}`, x, y, width, height }) as MoodboardItem

const CENTRE = { x: 50, y: 60 }

describe('placeImport', () => {
  it('lands at the viewport centre on an empty canvas', () => {
    const incoming = [{ id: 'a', x: 0, y: 0 }]
    expect(placeImport([], incoming, incoming[0], CENTRE)).toEqual(CENTRE)
  })

  it('lands a gutter clear of the right edge of what is already there', () => {
    const existing = [item(0, 0), item(400, 300)]
    const incoming = [{ id: 'a', x: 1000, y: 1000, width: 200 }]
    expect(placeImport(existing, incoming, incoming[0], CENTRE)).toEqual({
      x: 400 + 200 + IMPORT_GUTTER,
      y: 0,
    })
  })

  it('offsets the target by its position within the incoming graph', () => {
    const existing = [item(0, 0)]
    // The target sits down and to the right of the incoming graph's own top-left corner, and has
    // to keep that offset or the rest of the graph lands back over the existing nodes.
    const incoming = [
      { id: 'head', x: -500, y: -100, width: 200 },
      { id: 'target', x: -200, y: 40, width: 200 },
    ]
    const at = placeImport(existing, incoming, incoming[1], CENTRE)
    expect(at).toEqual({ x: 200 + IMPORT_GUTTER + 300, y: 140 })
  })

  it('measures the right edge from width, not from x alone', () => {
    const existing = [item(0, 0, 900)]
    const incoming = [{ id: 'a', x: 0, y: 0, width: 200 }]
    expect(placeImport(existing, incoming, incoming[0], CENTRE).x).toBe(900 + IMPORT_GUTTER)
  })

  it('falls back to the compact node width when an item declares none', () => {
    const existing = [{ id: 'x', x: 0, y: 0 } as MoodboardItem]
    const incoming = [{ id: 'a', x: 0, y: 0 }]
    expect(placeImport(existing, incoming, incoming[0], CENTRE).x).toBe(200 + IMPORT_GUTTER)
  })
})

describe('recipeTarget', () => {
  const items = [
    { id: 'a', x: 0, y: 0 },
    { id: 'b', x: 1, y: 1 },
  ]

  it('finds the named target', () => {
    expect(recipeTarget(items, 'b')).toBe(items[1])
  })

  it('falls back to the first item when the target is missing or unnamed', () => {
    expect(recipeTarget(items, 'nope')).toBe(items[0])
    expect(recipeTarget(items, undefined)).toBe(items[0])
  })
})

describe('boundsOf', () => {
  it('encloses every item, measuring the far edges from width and height', () => {
    expect(boundsOf([item(0, 0, 200, 120), item(400, 300, 100, 50)])).toEqual({
      x: 0,
      y: 0,
      width: 500,
      height: 350,
    })
  })

  it('falls back to the compact node size when an item declares none', () => {
    expect(boundsOf([{ x: 10, y: 20 }])).toEqual({
      x: 10,
      y: 20,
      width: 200,
      height: 120,
    })
  })

  it('is null for nothing, which is what makes an empty canvas fall back to the centre', () => {
    expect(boundsOf([])).toBeNull()
  })
})
