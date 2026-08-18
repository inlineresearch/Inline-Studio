import { describe, expect, it } from 'vitest'
import type { NodeDescriptor } from '@shared/coreNodes'
import { HALF_DOT, compactNodeMinHeight, portExtents, stackExtent } from './nodeSize'
import served from './nodeSize.fixture.json'

function make(
  inputs: [string, string][],
  outputs: [string, string][],
  params: [string, string][] = [],
): NodeDescriptor {
  return {
    type: 't',
    title: 'T',
    inputs: inputs.map(([id, kind]) => ({ id, label: id, kind })),
    outputs: outputs.map(([id, kind]) => ({ id, label: id, kind })),
    params: params.map(([key, widget]) => ({ key, label: key, widget, default: '' })),
  } as unknown as NodeDescriptor
}

describe('compactNodeMinHeight', () => {
  it('grows with the busier side, not the total', () => {
    // Dots stack down each edge independently, so three in and one out needs the same room as
    // three in and three out.
    const lopsided = make(
      [
        ['a', 'image'],
        ['b', 'image'],
        ['c', 'image'],
      ],
      [['o', 'image']],
    )
    const even = make(
      [
        ['a', 'image'],
        ['b', 'image'],
        ['c', 'image'],
      ],
      [
        ['o', 'image'],
        ['p', 'image'],
        ['q', 'image'],
      ],
    )
    expect(compactNodeMinHeight(lopsided)).toBe(compactNodeMinHeight(even))
    expect(compactNodeMinHeight(lopsided)).toBeGreaterThan(
      compactNodeMinHeight(make([['a', 'image']], [['o', 'image']])),
    )
  })

  it('adds room for the bottom-packed model band on top of the top-packed one', () => {
    // Content ports pack from the top and model ports from the bottom, so they need separate room
    // or the two stacks meet in the middle.
    const both = make(
      [
        ['image', 'image'],
        ['model', 'model'],
        ['vae', 'vae'],
      ],
      [['out', 'image']],
    )
    const { top, bottom } = portExtents(both)
    expect(bottom).toBeGreaterThan(0)
    expect(compactNodeMinHeight(both)).toBeGreaterThanOrEqual(top + bottom)
  })

  it('makes room for what the body renders: a row per dropdown', () => {
    const one = make([], [['o', 'image']], [['file', 'select']])
    const three = make(
      [],
      [['o', 'image']],
      [
        ['a', 'select'],
        ['b', 'select'],
        ['c', 'select'],
      ],
    )
    expect(compactNodeMinHeight(three)).toBeGreaterThan(compactNodeMinHeight(one))
  })

  it('makes room for the Adjust row when params sit behind it', () => {
    const plain = make([], [['o', 'image']], [['file', 'select']])
    const adjustable = make(
      [],
      [['o', 'image']],
      [
        ['file', 'select'],
        ['steps', 'number'],
      ],
    )
    expect(compactNodeMinHeight(adjustable)).toBeGreaterThan(compactNodeMinHeight(plain))
  })

  it('keeps every dot of every shipped compact node inside the node', () => {
    // The guarantee, against the descriptors Core actually serves rather than invented ones.
    const compact = (served as { models: NodeDescriptor[] }).models.filter(
      (d) => d.outputKind == null,
    )
    expect(compact.length).toBeGreaterThan(10)
    for (const descriptor of compact) {
      const height = compactNodeMinHeight(descriptor)
      const { top, bottom } = portExtents(descriptor)
      expect(top + bottom, `${descriptor.type} dots overflow`).toBeLessThanOrEqual(height)
    }
  })

  it('needs no room when a side has no dots', () => {
    expect(stackExtent(0)).toBe(0)
    expect(stackExtent(1)).toBeGreaterThan(HALF_DOT)
  })
})
