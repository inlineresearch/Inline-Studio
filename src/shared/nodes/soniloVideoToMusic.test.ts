import { describe, it, expect } from 'vitest'
import { SONILO_V2M } from './soniloVideoToMusic'
import { defaultParams, emptyResolvedInputs, type ResolvedInputs } from './types'

function withVideo(url: string): ResolvedInputs {
  return { ...emptyResolvedInputs(), videos: [url] }
}

describe('SONILO_V2M.resolveEndpoint', () => {
  it('always targets the single video-to-music endpoint', () => {
    expect(SONILO_V2M.resolveEndpoint(emptyResolvedInputs())).toBe('sonilo/v1.1/video-to-music')
  })
})

describe('SONILO_V2M.buildRequest', () => {
  it('maps the first resolved video to video_url and omits the optional fields at defaults', () => {
    const body = SONILO_V2M.buildRequest(
      { ...defaultParams(SONILO_V2M), prompt: '' },
      withVideo('https://fal/cut.mp4'),
    )
    expect(body).toEqual({ video_url: 'https://fal/cut.mp4', num_samples: 1 })
    expect(body.prompt).toBeUndefined()
    expect(body.start_offset).toBeUndefined()
    expect(body.duration).toBeUndefined()
  })

  it('passes a wired style prompt through, trimmed', () => {
    const body = SONILO_V2M.buildRequest(
      { ...defaultParams(SONILO_V2M), prompt: '  warm analog synths  ' },
      withVideo('u'),
    )
    expect(body.prompt).toBe('warm analog synths')
  })

  it('includes a positive segment start/duration and coerces the sample count', () => {
    const body = SONILO_V2M.buildRequest(
      { ...defaultParams(SONILO_V2M), num_samples: 3, start_offset: 12, duration: 30 },
      withVideo('u'),
    )
    expect(body).toMatchObject({ num_samples: 3, start_offset: 12, duration: 30 })
    expect(
      SONILO_V2M.buildRequest({ ...defaultParams(SONILO_V2M), num_samples: 0 }, withVideo('u'))
        .num_samples,
    ).toBe(1)
  })
})

describe('SONILO_V2M.parseOutputs', () => {
  it('maps the audios array to m4a refs', () => {
    const refs = SONILO_V2M.parseOutputs({
      audio: { url: 'https://fal/a.m4a', content_type: 'audio/mp4' },
      audios: [
        { url: 'https://fal/a.m4a', content_type: 'audio/mp4' },
        { url: 'https://fal/b.m4a', content_type: 'audio/mp4' },
      ],
    })
    expect(refs).toEqual([
      { url: 'https://fal/a.m4a', ext: '.m4a', kind: 'audio' },
      { url: 'https://fal/b.m4a', ext: '.m4a', kind: 'audio' },
    ])
  })

  it('returns [] when the audios are missing or malformed', () => {
    expect(SONILO_V2M.parseOutputs({})).toEqual([])
    expect(SONILO_V2M.parseOutputs({ audios: [{ url: '' }] })).toEqual([])
    expect(SONILO_V2M.parseOutputs(null)).toEqual([])
  })
})

describe('SONILO_V2M.estimatePrice', () => {
  it('prices per second × samples once a segment duration is set', () => {
    // 30 s × 2 samples × $0.009/s = $0.54.
    const est = SONILO_V2M.estimatePrice?.({ duration: 30, num_samples: 2 })
    expect(est?.amount).toBeCloseTo(0.54, 5)
    expect(est?.approx).toBe(true)
  })

  it('returns null at the full-video default (the video length is unknown here)', () => {
    expect(SONILO_V2M.estimatePrice?.(defaultParams(SONILO_V2M))).toBeNull()
  })
})

describe('SONILO_V2M shape', () => {
  it('declares a required video input, an optional prompt, and an audio output', () => {
    expect(SONILO_V2M.inputs).toHaveLength(1)
    expect(SONILO_V2M.inputs[0]).toMatchObject({ id: 'video', kind: 'video', required: true })
    expect(SONILO_V2M.promptOptional).toBe(true)
    expect(SONILO_V2M.outputKind).toBe('audio')
  })
})
