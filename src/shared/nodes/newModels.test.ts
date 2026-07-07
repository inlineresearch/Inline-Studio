import { describe, it, expect } from 'vitest'
import { emptyResolvedInputs, type ResolvedInputs } from './types'
import { NANO_BANANA_2 } from './nanoBanana2'
import { NANO_BANANA_PRO } from './nanoBananaPro'
import { KREA_V2 } from './kreaV2'
import { SEEDANCE_T2V } from './seedanceT2V'
import { SEEDANCE_I2V } from './seedanceI2V'
import { SEEDANCE_REF2V } from './seedanceRef2V'

const withImages = (images: string[]): ResolvedInputs => ({ ...emptyResolvedInputs(), images })

describe('nano-banana-2 (text→image)', () => {
  it('constant endpoint + text-to-image body', () => {
    expect(NANO_BANANA_2.resolveEndpoint(emptyResolvedInputs())).toBe('fal-ai/nano-banana-2')
    const body = NANO_BANANA_2.buildRequest(
      { prompt: 'a cat', resolution: '2K', limit_generations: 3 },
      emptyResolvedInputs(),
    )
    expect(body).toMatchObject({
      prompt: 'a cat',
      resolution: '2K',
      limit_generations: 3,
      aspect_ratio: 'auto',
    })
  })
  it('prices per image × resolution tier × count', () => {
    // 0.08 × 2 (4K) × 2 = 0.32
    expect(
      NANO_BANANA_2.estimatePrice?.({ resolution: '4K', limit_generations: 2 })?.amount,
    ).toBeCloseTo(0.32, 5)
  })
  it('parses images', () => {
    expect(
      NANO_BANANA_2.parseOutputs({
        images: [{ url: 'https://f/a.png', content_type: 'image/png' }],
      }),
    ).toEqual([{ url: 'https://f/a.png', ext: '.png', kind: 'image' }])
  })
})

describe('nano-banana-pro/edit (image→image)', () => {
  it('requires an image input and passes image_urls', () => {
    expect(NANO_BANANA_PRO.inputs[0]).toMatchObject({ kind: 'image[]', required: true })
    const body = NANO_BANANA_PRO.buildRequest(
      { prompt: 'edit' },
      withImages(['data:img1', 'data:img2']),
    )
    expect(body).toMatchObject({ prompt: 'edit', image_urls: ['data:img1', 'data:img2'] })
  })
  it('4K doubles the per-image price', () => {
    expect(
      NANO_BANANA_PRO.estimatePrice?.({ resolution: '4K', num_images: 1 })?.amount,
    ).toBeCloseTo(0.3, 5)
  })
})

describe('krea v2 large (text→image)', () => {
  it('omits a negative seed, includes a set one', () => {
    expect(KREA_V2.resolveEndpoint(emptyResolvedInputs())).toBe('krea/v2/large/text-to-image')
    expect(
      KREA_V2.buildRequest({ prompt: 'p', seed: -1 }, emptyResolvedInputs()).seed,
    ).toBeUndefined()
    expect(KREA_V2.buildRequest({ prompt: 'p', seed: 7 }, emptyResolvedInputs()).seed).toBe(7)
  })
})

describe('seedance 2.0 (video)', () => {
  it('text-to-video: shared body, no image', () => {
    expect(SEEDANCE_T2V.resolveEndpoint(emptyResolvedInputs())).toBe(
      'bytedance/seedance-2.0/text-to-video',
    )
    const body = SEEDANCE_T2V.buildRequest(
      { prompt: 'p', resolution: '1080p' },
      emptyResolvedInputs(),
    )
    expect(body).toMatchObject({
      prompt: 'p',
      resolution: '1080p',
      duration: 'auto',
      generate_audio: true,
    })
    expect(body.image_url).toBeUndefined()
  })
  it('image-to-video: passes image_url', () => {
    const body = SEEDANCE_I2V.buildRequest({ prompt: 'p' }, withImages(['data:frame']))
    expect(body).toMatchObject({ image_url: 'data:frame' })
  })
  it('reference-to-video: maps image/video/audio buckets and declares an audio input', () => {
    expect(SEEDANCE_REF2V.inputs.some((p) => p.kind === 'audio')).toBe(true)
    const body = SEEDANCE_REF2V.buildRequest(
      { prompt: '@Image1 dances to @Audio1' },
      { ...emptyResolvedInputs(), images: ['i1'], videos: ['v1'], audios: ['a1'] },
    )
    expect(body).toMatchObject({ image_urls: ['i1'], video_urls: ['v1'], audio_urls: ['a1'] })
  })
  it('prices per second by resolution/duration', () => {
    // 720p ≈ $0.30/s × 6s = 1.80
    expect(SEEDANCE_T2V.estimatePrice?.({ resolution: '720p', duration: '6' })?.amount).toBeCloseTo(
      1.8,
      5,
    )
  })
})
