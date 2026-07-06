import { describe, it, expect } from 'vitest'
import { GPT_IMAGE_2 } from './gptImage2'
import { defaultParams, emptyResolvedInputs, type ResolvedInputs } from './types'

function withImages(images: string[], masks: string[] = []): ResolvedInputs {
  return { ...emptyResolvedInputs(), images, masks }
}

describe('GPT_IMAGE_2.resolveEndpoint', () => {
  it('uses the single base endpoint for text-to-image', () => {
    expect(GPT_IMAGE_2.resolveEndpoint(emptyResolvedInputs())).toBe('openai/gpt-image-2')
  })

  it('uses the same base endpoint when images are wired (editing via image_urls, no sub-path)', () => {
    expect(GPT_IMAGE_2.resolveEndpoint(withImages(['https://fal/img.png']))).toBe(
      'openai/gpt-image-2',
    )
  })
})

describe('GPT_IMAGE_2.buildRequest', () => {
  it('maps default params to the request body and omits image_urls for text-to-image', () => {
    const body = GPT_IMAGE_2.buildRequest(defaultParams(GPT_IMAGE_2), emptyResolvedInputs())
    expect(body).toEqual({
      prompt: '',
      image_size: 'landscape_4_3',
      quality: 'high',
      num_images: 1,
      output_format: 'png',
    })
    expect(body.image_urls).toBeUndefined()
  })

  it('includes image_urls when images are wired', () => {
    const body = GPT_IMAGE_2.buildRequest(
      { ...defaultParams(GPT_IMAGE_2), prompt: 'a cat' },
      withImages(['https://fal/a.png', 'https://fal/b.png']),
    )
    expect(body.prompt).toBe('a cat')
    expect(body.image_urls).toEqual(['https://fal/a.png', 'https://fal/b.png'])
    expect(body.mask_url).toBeUndefined()
  })

  it('includes mask_url only when a mask is resolved', () => {
    const body = GPT_IMAGE_2.buildRequest(
      defaultParams(GPT_IMAGE_2),
      withImages(['https://fal/a.png'], ['https://fal/mask.png']),
    )
    expect(body.mask_url).toBe('https://fal/mask.png')
  })

  it('coerces num_images to a number', () => {
    const body = GPT_IMAGE_2.buildRequest(
      { ...defaultParams(GPT_IMAGE_2), num_images: 4 },
      emptyResolvedInputs(),
    )
    expect(body.num_images).toBe(4)
  })
})

describe('GPT_IMAGE_2.parseOutputs', () => {
  it('maps a single png image', () => {
    const refs = GPT_IMAGE_2.parseOutputs({
      images: [{ url: 'https://fal/out.png', content_type: 'image/png' }],
    })
    expect(refs).toEqual([{ url: 'https://fal/out.png', ext: '.png', kind: 'image' }])
  })

  it('maps multiple images and derives ext from content type', () => {
    const refs = GPT_IMAGE_2.parseOutputs({
      images: [
        { url: 'https://fal/a.webp', content_type: 'image/webp' },
        { url: 'https://fal/b.jpg', content_type: 'image/jpeg' },
      ],
    })
    expect(refs.map((r) => r.ext)).toEqual(['.webp', '.jpg'])
  })

  it('returns [] for an empty or malformed response', () => {
    expect(GPT_IMAGE_2.parseOutputs({})).toEqual([])
    expect(GPT_IMAGE_2.parseOutputs({ images: [] })).toEqual([])
    expect(GPT_IMAGE_2.parseOutputs({ images: [{ url: '' }] })).toEqual([])
    expect(GPT_IMAGE_2.parseOutputs(null)).toEqual([])
  })
})

describe('GPT_IMAGE_2.estimatePrice', () => {
  it('prices per image by quality × count', () => {
    expect(GPT_IMAGE_2.estimatePrice?.({ quality: 'high', num_images: 1 })).toEqual({
      amount: 0.145,
      approx: true,
    })
    expect(GPT_IMAGE_2.estimatePrice?.({ quality: 'low', num_images: 4 })?.amount).toBeCloseTo(0.02)
    expect(GPT_IMAGE_2.estimatePrice?.({ quality: 'medium', num_images: 2 })?.amount).toBeCloseTo(
      0.074,
    )
  })

  it('defaults to high quality × one image', () => {
    expect(GPT_IMAGE_2.estimatePrice?.({})?.amount).toBeCloseTo(0.145)
  })
})

describe('defaultParams(GPT_IMAGE_2)', () => {
  it('fills every declared param field', () => {
    const params = defaultParams(GPT_IMAGE_2)
    for (const field of GPT_IMAGE_2.params) {
      expect(params[field.key]).toBe(field.default)
    }
  })
})
