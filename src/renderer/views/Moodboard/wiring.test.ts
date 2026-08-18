import { describe, expect, it } from 'vitest'
import type { NodeDescriptor } from '@shared/coreNodes'
import type { MoodboardItem } from '@shared/types'
import { canWire } from './wiring'

const DESCRIPTORS = [
  {
    type: 'character/encode',
    inputs: [
      { id: 'images', label: 'References', kind: 'image[]' },
      { id: 'description', label: 'Description', kind: 'text' },
    ],
    outputs: [{ id: 'character', label: 'Character', kind: 'character' }],
  },
  {
    type: 'alibaba/z-image-turbo',
    inputs: [
      { id: 'prompt', label: 'Prompt', kind: 'text' },
      { id: 'image', label: 'Image', kind: 'image' },
    ],
    outputs: [{ id: 'image', label: 'Image', kind: 'image' }],
  },
] as unknown as NodeDescriptor[]

function item(id: string, type: string, coreType?: string): MoodboardItem {
  return {
    id,
    type,
    data: coreType ? { core: { type: coreType, params: {} } } : {},
  } as MoodboardItem
}

const ITEMS = [
  item('p', 'prompt'),
  item('loader', 'loader'),
  item('enc', 'core', 'character/encode'),
  item('gen', 'core', 'alibaba/z-image-turbo'),
  item('space', 'controlSpace'),
]

describe('canWire', () => {
  it('lets a Prompt feed a text input whatever the handle is called', () => {
    // The rule used to key off the handle name 'prompt', so Encode Character's `description`
    // input silently refused every wire.
    expect(
      canWire(
        { source: 'p', target: 'enc', sourceHandle: 'out', targetHandle: 'description' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(true)
    expect(
      canWire(
        { source: 'p', target: 'gen', sourceHandle: 'out', targetHandle: 'prompt' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(true)
  })

  it('still refuses a Prompt into a media input', () => {
    expect(
      canWire(
        { source: 'p', target: 'enc', sourceHandle: 'out', targetHandle: 'images' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(false)
    expect(
      canWire(
        { source: 'p', target: 'gen', sourceHandle: 'out', targetHandle: 'image' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(false)
  })

  it('refuses media into a text input', () => {
    expect(
      canWire(
        { source: 'loader', target: 'enc', sourceHandle: 'out', targetHandle: 'description' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(false)
  })

  it('allows an image source into an image list', () => {
    expect(
      canWire(
        { source: 'loader', target: 'enc', sourceHandle: 'out', targetHandle: 'images' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(true)
  })

  it('keeps a Control Space out of the plain image input', () => {
    // It emits a pose map; wired to img2img the pose would be ignored and the render look fine.
    expect(
      canWire(
        { source: 'space', target: 'gen', sourceHandle: 'out', targetHandle: 'image' },
        ITEMS,
        DESCRIPTORS,
      ),
    ).toBe(false)
  })

  it('refuses a node wired to itself, and half a wire', () => {
    expect(canWire({ source: 'gen', target: 'gen' }, ITEMS, DESCRIPTORS)).toBe(false)
    expect(canWire({ source: 'p', target: null }, ITEMS, DESCRIPTORS)).toBe(false)
  })
})
