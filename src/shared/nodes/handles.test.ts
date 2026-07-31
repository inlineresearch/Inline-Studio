import { describe, expect, it } from 'vitest'
import { dotPorts, handleIdForPort, portIdForHandle } from './handles'
import { MINIMAX_H3_I2V } from './minimaxH3I2V'
import { MINIMAX_H3_REF2V } from './minimaxH3Ref2V'
import { SEEDANCE_I2V } from './seedanceI2V'
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

  it('orders dots media-first, then audio', () => {
    expect(dotPorts(MINIMAX_H3_REF2V).map((p) => p.id)).toEqual([
      'reference_image_urls',
      'reference_video_urls',
      'reference_audio_urls',
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
