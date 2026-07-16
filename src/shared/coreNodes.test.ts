import { describe, expect, it } from 'vitest'
import { isEngineKind, isModelPort, portKindColor, portsSatisfy, producesFrame } from './coreNodes'
import type { NodeDescriptor } from './coreNodes'

describe('portsSatisfy', () => {
  it('accepts an exact kind match', () => {
    expect(portsSatisfy('model', 'model')).toBe(true)
    expect(portsSatisfy('latent', 'latent')).toBe(true)
  })
  it('accepts a single image into an image list, not the reverse', () => {
    expect(portsSatisfy('image', 'image[]')).toBe(true)
    expect(portsSatisfy('image[]', 'image')).toBe(false)
  })
  it('rejects mismatched engine kinds', () => {
    expect(portsSatisfy('vae', 'model')).toBe(false)
    expect(portsSatisfy('latent', 'image')).toBe(false)
  })
})

describe('kind helpers', () => {
  it('classifies engine vs media kinds', () => {
    expect(isEngineKind('model')).toBe(true)
    expect(isEngineKind('conditioning')).toBe(true)
    expect(isEngineKind('image')).toBe(false)
  })
  it('gives every kind a color', () => {
    expect(portKindColor('model')).toMatch(/^#/)
    expect(portKindColor('image')).toMatch(/^#/)
  })
  it('treats only model/vae/text-encoder as model-family ports (bottom-packed dots)', () => {
    expect(isModelPort('model')).toBe(true)
    expect(isModelPort('vae')).toBe(true)
    expect(isModelPort('text-encoder')).toBe(true)
    // Signal/content kinds — including the other engine kinds — pack to the top instead.
    expect(isModelPort('conditioning')).toBe(false)
    expect(isModelPort('latent')).toBe(false)
    expect(isModelPort('image')).toBe(false)
  })
})

describe('producesFrame', () => {
  const base: NodeDescriptor = {
    type: 't',
    title: 'T',
    category: 'C',
    icon: '',
    source: 'builtin',
    outputKind: null,
    inputs: [],
    outputs: [],
    params: [],
  }
  it('is true only for media outputs', () => {
    expect(producesFrame({ ...base, outputKind: 'image' })).toBe(true)
    expect(producesFrame(base)).toBe(false)
  })
})
