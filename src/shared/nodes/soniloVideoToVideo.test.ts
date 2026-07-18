import { describe, it, expect } from 'vitest'
import { SONILO_V2V } from './soniloVideoToVideo'
import { defaultParams, emptyResolvedInputs, type ResolvedInputs } from './types'

function withVideo(url: string): ResolvedInputs {
  return { ...emptyResolvedInputs(), videos: [url] }
}

describe('SONILO_V2V.resolveEndpoint', () => {
  it('always targets the single video-to-video endpoint', () => {
    expect(SONILO_V2V.resolveEndpoint(emptyResolvedInputs())).toBe('sonilo/v1.1/video-to-video')
  })
})

describe('SONILO_V2V.buildRequest', () => {
  it('maps the video to video_url and keeps original speech off by default', () => {
    const body = SONILO_V2V.buildRequest(
      { ...defaultParams(SONILO_V2V), prompt: '' },
      withVideo('https://fal/cut.mp4'),
    )
    expect(body).toEqual({
      video_url: 'https://fal/cut.mp4',
      num_samples: 1,
      keep_speech_vocal: false,
    })
    expect(body.prompt).toBeUndefined()
    expect(body.start_offset).toBeUndefined()
    expect(body.duration).toBeUndefined()
  })

  it('passes the keep-speech toggle and a trimmed style prompt through', () => {
    const body = SONILO_V2V.buildRequest(
      { ...defaultParams(SONILO_V2V), keep_speech_vocal: true, prompt: '  warm analog synths  ' },
      withVideo('u'),
    )
    expect(body.keep_speech_vocal).toBe(true)
    expect(body.prompt).toBe('warm analog synths')
  })

  it('includes a positive segment start/duration and coerces the sample count', () => {
    const body = SONILO_V2V.buildRequest(
      { ...defaultParams(SONILO_V2V), num_samples: 3, start_offset: 12, duration: 30 },
      withVideo('u'),
    )
    expect(body).toMatchObject({ num_samples: 3, start_offset: 12, duration: 30 })
    expect(
      SONILO_V2V.buildRequest({ ...defaultParams(SONILO_V2V), num_samples: 0 }, withVideo('u'))
        .num_samples,
    ).toBe(1)
  })
})

describe('SONILO_V2V.estimatePrice', () => {
  it('prices per second × samples once a segment duration is set', () => {
    // 30 s × 2 samples × $0.009/s = $0.54.
    const est = SONILO_V2V.estimatePrice?.({ duration: 30, num_samples: 2 })
    expect(est?.amount).toBeCloseTo(0.54, 5)
    expect(est?.approx).toBe(true)
  })

  it('returns null at the full-video default (the video length is unknown here)', () => {
    expect(SONILO_V2V.estimatePrice?.(defaultParams(SONILO_V2V))).toBeNull()
  })
})

describe('SONILO_V2V shape', () => {
  it('declares a required video input, an optional prompt, and a video output', () => {
    expect(SONILO_V2V.inputs).toHaveLength(1)
    expect(SONILO_V2V.inputs[0]).toMatchObject({ id: 'video', kind: 'video', required: true })
    expect(SONILO_V2V.promptOptional).toBe(true)
    expect(SONILO_V2V.outputKind).toBe('video')
    expect(SONILO_V2V.provider).toBe('fal')
  })
})
