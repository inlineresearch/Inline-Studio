import { describe, expect, it } from 'vitest'
import type { NodeDescriptor } from '@shared/coreNodes'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import { wiredParams } from './wiredParams'

const ENCODE = {
  type: 'character/encode',
  inputs: [
    { id: 'images', label: 'References', kind: 'image[]' },
    { id: 'description', label: 'Description', kind: 'text' },
  ],
  outputs: [],
  params: [
    { key: 'name', label: 'Name', widget: 'text', default: '' },
    { key: 'description', label: 'Description', widget: 'textarea', default: '' },
  ],
} as unknown as NodeDescriptor

const GEN = {
  type: 'alibaba/z-image-turbo',
  inputs: [{ id: 'model', label: 'Model', kind: 'model' }],
  outputs: [],
  params: [{ key: 'model', label: 'Model', widget: 'select', default: '' }],
} as unknown as NodeDescriptor

function prompt(id: string, text: string): MoodboardItem {
  return { id, type: 'prompt', data: { promptText: text } } as MoodboardItem
}
function core(id: string, type: string, params: Record<string, unknown> = {}): MoodboardItem {
  return { id, type: 'core', data: { core: { type, params } } } as MoodboardItem
}
function wire(from: string, to: string, targetHandle: string): MoodboardConnector {
  return {
    id: `${from}-${to}`,
    fromItemId: from,
    toItemId: to,
    data: { targetHandle },
  } as unknown as MoodboardConnector
}

describe('wiredParams', () => {
  it('reports a Prompt driving a same-named text param', () => {
    const items = [prompt('p', 'a lighthouse at dusk'), core('enc', 'character/encode')]
    const found = wiredParams('enc', ENCODE, items, [wire('p', 'enc', 'description')])
    expect(found.get('description')).toEqual({
      from: 'Prompt',
      text: 'a lighthouse at dusk',
      fallsBack: false,
    })
  })

  it('says the typed value still applies when the wired Prompt is empty', () => {
    // The runner reads `wired or typed`, so an empty Prompt is not an override.
    const items = [prompt('p', '   '), core('enc', 'character/encode')]
    const found = wiredParams('enc', ENCODE, items, [wire('p', 'enc', 'description')])
    expect(found.get('description')?.fallsBack).toBe(true)
  })

  it('treats an empty component handle as an override anyway', () => {
    // `component_ref` only returns null when nothing is wired, so a wired loader always wins.
    const items = [core('l', 'load/diffusion-model'), core('gen', 'alibaba/z-image-turbo')]
    const found = wiredParams('gen', GEN, items, [wire('l', 'gen', 'model')])
    expect(found.get('model')?.fallsBack).toBe(false)
  })

  it('resolves a loader to the file it names', () => {
    const items = [
      core('l', 'load/diffusion-model', { file: 'z_image_bf16.safetensors' }),
      core('gen', 'alibaba/z-image-turbo'),
    ]
    const found = wiredParams('gen', GEN, items, [wire('l', 'gen', 'model')])
    expect(found.get('model')?.text).toBe('z_image_bf16.safetensors')
  })

  it('ignores wires into ports that are not also params', () => {
    const items = [core('src', 'x'), core('enc', 'character/encode')]
    const found = wiredParams('enc', ENCODE, items, [wire('src', 'enc', 'images')])
    expect(found.size).toBe(0)
  })

  it('ignores wires into other nodes, and a missing descriptor', () => {
    const items = [prompt('p', 'x'), core('enc', 'character/encode')]
    expect(wiredParams('other', ENCODE, items, [wire('p', 'enc', 'description')]).size).toBe(0)
    expect(wiredParams('enc', undefined, items, [wire('p', 'enc', 'description')]).size).toBe(0)
  })
})
