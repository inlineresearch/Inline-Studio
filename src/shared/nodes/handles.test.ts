import { describe, expect, it } from 'vitest'
import { dotPorts, handleIdForPort, portIdForHandle } from './handles'
import { MINIMAX_H3_I2V } from './minimaxH3I2V'
import { MINIMAX_H3_REF2V } from './minimaxH3Ref2V'
import { SEEDANCE_I2V } from './seedanceI2V'
import { SEEDANCE_REF2V } from './seedanceRef2V'
import { SONILO_V2M } from './soniloVideoToMusic'
import { emptyResolvedInputs, portMedia, type NodeDef, type ResolvedInputs } from './types'

describe('canvas handle ids', () => {
  it('keeps the legacy `in` / `audio` ids for the first port of each kind', () => {
    // Renaming these would orphan every connector in every existing project.
    expect(handleIdForPort(SEEDANCE_I2V, SEEDANCE_I2V.inputs[0])).toBe('in')
    expect(handleIdForPort(SONILO_V2M, SONILO_V2M.inputs[0])).toBe('in')
    const [images, video, audio] = MINIMAX_H3_REF2V.inputs
    expect(handleIdForPort(MINIMAX_H3_REF2V, images)).toBe('in')
    expect(handleIdForPort(MINIMAX_H3_REF2V, video)).toBe('reference_video_urls')
    expect(handleIdForPort(MINIMAX_H3_REF2V, audio)).toBe('audio')
  })

  it('gives a second port of the same kind its own id', () => {
    const [start, end] = MINIMAX_H3_I2V.inputs
    expect(handleIdForPort(MINIMAX_H3_I2V, start)).toBe('in')
    expect(handleIdForPort(MINIMAX_H3_I2V, end)).toBe('end_image')
  })

  it('resolves a handle back to the port it means', () => {
    expect(portIdForHandle(MINIMAX_H3_I2V, 'in')).toBe('image')
    expect(portIdForHandle(MINIMAX_H3_I2V, 'end_image')).toBe('end_image')
    expect(portIdForHandle(MINIMAX_H3_I2V, 'prompt')).toBeNull()
    expect(portIdForHandle(MINIMAX_H3_I2V, undefined)).toBeNull()
  })

  it('orders dots media-first, then audio, then character', () => {
    expect(dotPorts(MINIMAX_H3_REF2V).map((p) => p.id)).toEqual([
      'reference_image_urls',
      'reference_video_urls',
      'reference_audio_urls',
      'character',
    ])
  })
})

describe('portMedia', () => {
  const twoImagePorts: NodeDef = MINIMAX_H3_I2V

  it('prefers an explicit wire over the kind bucket', () => {
    const resolved: ResolvedInputs = {
      ...emptyResolvedInputs(),
      images: ['data:a', 'data:b'],
      byHandle: { end_image: ['data:b'] },
    }
    expect(portMedia(twoImagePorts, resolved, 'end_image')).toEqual(['data:b'])
    // 'a' is the only unclaimed image, so it fills the start port.
    expect(portMedia(twoImagePorts, resolved, 'image')).toEqual(['data:a'])
  })

  it('never hands an explicitly-wired item to another port', () => {
    const resolved: ResolvedInputs = {
      ...emptyResolvedInputs(),
      images: ['data:only'],
      byHandle: { end_image: ['data:only'] },
    }
    expect(portMedia(twoImagePorts, resolved, 'image')).toEqual([])
  })

  it('falls back positionally when nothing is tagged', () => {
    const resolved: ResolvedInputs = { ...emptyResolvedInputs(), images: ['data:a', 'data:b'] }
    expect(portMedia(twoImagePorts, resolved, 'image')).toEqual(['data:a'])
    expect(portMedia(twoImagePorts, resolved, 'end_image')).toEqual(['data:b'])
  })

  it('gives a list port the whole untagged bucket', () => {
    const resolved: ResolvedInputs = {
      ...emptyResolvedInputs(),
      images: ['data:a', 'data:b', 'data:c'],
    }
    expect(portMedia(MINIMAX_H3_REF2V, resolved, 'reference_image_urls')).toEqual([
      'data:a',
      'data:b',
      'data:c',
    ])
  })

  it('is unchanged for a def with a single port of a kind', () => {
    const resolved: ResolvedInputs = { ...emptyResolvedInputs(), images: ['data:a'] }
    expect(portMedia(SEEDANCE_I2V, resolved, 'image')).toEqual(['data:a'])
  })

  it('returns nothing for an unknown port', () => {
    expect(portMedia(SEEDANCE_I2V, emptyResolvedInputs(), 'nope')).toEqual([])
  })
})

describe('the character dot', () => {
  it('renders last, after every media and audio port', () => {
    // Placed after the media dots so adding it never shifts the position of a dot a user already
    // knows, and so the strip still reads media-then-audio top to bottom.
    expect(dotPorts(SEEDANCE_REF2V).map((p) => p.id)).toEqual([
      'image_urls',
      'video_urls',
      'audio_urls',
      'character',
    ])
  })

  it('keeps its own handle id, never a legacy one', () => {
    // It arrived long after `in`/`audio` were fixed, so it has no legacy id to inherit - and it
    // must not take `in`, which every existing media connector already stores.
    const character = SEEDANCE_REF2V.inputs.find((p) => p.kind === 'character')
    expect(character && handleIdForPort(SEEDANCE_REF2V, character)).toBe('character')
    expect(portIdForHandle(SEEDANCE_REF2V, 'character')).toBe('character')
    expect(handleIdForPort(SEEDANCE_REF2V, SEEDANCE_REF2V.inputs[0])).toBe('in')
  })

  it('draws no media, so it never claims a resolved input', () => {
    // `mediaFamily` is null for it, which is what keeps every media accessor inert.
    const resolved: ResolvedInputs = {
      ...emptyResolvedInputs(),
      images: ['data:image/png;base64,a'],
    }
    expect(portMedia(SEEDANCE_REF2V, resolved, 'character')).toEqual([])
  })
})
