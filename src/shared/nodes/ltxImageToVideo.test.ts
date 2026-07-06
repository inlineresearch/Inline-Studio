import { describe, it, expect } from 'vitest'
import { LTX_I2V } from './ltxImageToVideo'
import { defaultParams, emptyResolvedInputs, type ResolvedInputs } from './types'

function withImage(url: string): ResolvedInputs {
  return { ...emptyResolvedInputs(), images: [url] }
}

describe('LTX_I2V.resolveEndpoint', () => {
  it('always targets the single image-to-video endpoint', () => {
    expect(LTX_I2V.resolveEndpoint(emptyResolvedInputs())).toBe(
      'fal-ai/ltx-2.3-quality/image-to-video',
    )
  })
})

describe('LTX_I2V.buildRequest', () => {
  it('maps the first resolved image to image_url and omits a negative seed', () => {
    const body = LTX_I2V.buildRequest(defaultParams(LTX_I2V), withImage('https://fal/frame.png'))
    expect(body).toEqual({
      prompt: '',
      image_url: 'https://fal/frame.png',
      number_of_frames: 96,
      resolution: 'landscape_16_9',
      generate_audio: false,
    })
    expect(body.seed).toBeUndefined()
  })

  it('migrates a legacy "W x H" resolution to a valid preset by orientation', () => {
    const res = (v: string): unknown =>
      LTX_I2V.buildRequest({ ...defaultParams(LTX_I2V), resolution: v }, withImage('u')).resolution
    expect(res('1280 x 720')).toBe('landscape_16_9')
    expect(res('720 x 1280')).toBe('portrait_16_9')
    expect(res('1024 x 1024')).toBe('square_hd')
    expect(res('portrait_4_3')).toBe('portrait_4_3') // an already-valid preset passes through
  })

  it('includes a non-negative seed', () => {
    const body = LTX_I2V.buildRequest(
      { ...defaultParams(LTX_I2V), seed: 504804082 },
      withImage('https://fal/frame.png'),
    )
    expect(body.seed).toBe(504804082)
  })

  it('coerces generate_audio to a boolean', () => {
    const body = LTX_I2V.buildRequest(
      { ...defaultParams(LTX_I2V), generate_audio: true },
      withImage('https://fal/frame.png'),
    )
    expect(body.generate_audio).toBe(true)
  })
})

describe('LTX_I2V.parseOutputs', () => {
  it('maps the video object to a single mp4 ref', () => {
    const refs = LTX_I2V.parseOutputs({
      video: { url: 'https://fal/out.mp4', content_type: 'video/mp4' },
      seed: 1,
    })
    expect(refs).toEqual([{ url: 'https://fal/out.mp4', ext: '.mp4', kind: 'video' }])
  })

  it('returns [] when the video is missing or malformed', () => {
    expect(LTX_I2V.parseOutputs({})).toEqual([])
    expect(LTX_I2V.parseOutputs({ video: { url: '' } })).toEqual([])
    expect(LTX_I2V.parseOutputs(null)).toEqual([])
  })
})

describe('LTX_I2V.estimatePrice', () => {
  it('prices per megapixel = preset width × height × frames × rate', () => {
    // landscape_16_9 = 1024×576; ×121 frames ≈ 71 MP × $0.0024075 ≈ $0.17.
    const est = LTX_I2V.estimatePrice?.({ resolution: 'landscape_16_9', number_of_frames: 121 })
    expect(est?.amount).toBeCloseTo(0.17, 2)
    expect(est?.approx).toBe(true)
  })

  it('scales with frame count', () => {
    const a =
      LTX_I2V.estimatePrice?.({ resolution: 'landscape_16_9', number_of_frames: 48 })?.amount ?? 0
    const b =
      LTX_I2V.estimatePrice?.({ resolution: 'landscape_16_9', number_of_frames: 96 })?.amount ?? 0
    expect(b).toBeCloseTo(a * 2, 5)
  })
})

describe('LTX_I2V shape', () => {
  it('declares a required image input and a video output', () => {
    expect(LTX_I2V.inputs).toHaveLength(1)
    expect(LTX_I2V.inputs[0]).toMatchObject({ id: 'image', kind: 'image', required: true })
    expect(LTX_I2V.outputKind).toBe('video')
  })
})
