import { describe, it, expect } from 'vitest'
import { SONILO_T2M } from './soniloTextToMusic'
import { defaultParams, emptyResolvedInputs } from './types'

describe('SONILO_T2M.resolveEndpoint', () => {
  it('always targets the single text-to-music endpoint', () => {
    expect(SONILO_T2M.resolveEndpoint(emptyResolvedInputs())).toBe('sonilo/v1.1/text-to-music')
  })
})

describe('SONILO_T2M.buildRequest', () => {
  it('sends the trimmed prompt with the duration and sample defaults', () => {
    const body = SONILO_T2M.buildRequest(
      { ...defaultParams(SONILO_T2M), prompt: '  warm analog synths  ' },
      emptyResolvedInputs(),
    )
    expect(body).toEqual({ prompt: 'warm analog synths', duration: 90, num_samples: 1 })
  })

  it('clamps the duration to the 600 s ceiling and coerces the sample count up to 1', () => {
    const body = SONILO_T2M.buildRequest(
      { prompt: 'x', duration: 5000, num_samples: 0 },
      emptyResolvedInputs(),
    )
    expect(body).toMatchObject({ duration: 600, num_samples: 1 })
  })

  it('falls back to the 90 s default for a non-positive or unparseable duration', () => {
    expect(
      SONILO_T2M.buildRequest({ prompt: 'x', duration: 0 }, emptyResolvedInputs()).duration,
    ).toBe(90)
    expect(
      SONILO_T2M.buildRequest({ prompt: 'x', duration: 'soon' }, emptyResolvedInputs()).duration,
    ).toBe(90)
  })
})

describe('SONILO_T2M.estimatePrice', () => {
  it('prices per second × samples (duration is always known for text-to-music)', () => {
    // 120 s × 2 samples × $0.009/s = $2.16.
    const est = SONILO_T2M.estimatePrice?.({ duration: 120, num_samples: 2 })
    expect(est?.amount).toBeCloseTo(2.16, 5)
    expect(est?.approx).toBe(true)
  })

  it('prices the defaults (90 s × 1 sample = $0.81)', () => {
    const est = SONILO_T2M.estimatePrice?.(defaultParams(SONILO_T2M))
    expect(est?.amount).toBeCloseTo(0.81, 5)
  })
})

describe('SONILO_T2M shape', () => {
  it('takes no media input, keeps the prompt required, and outputs audio', () => {
    expect(SONILO_T2M.inputs).toHaveLength(0)
    expect(SONILO_T2M.promptOptional).toBeUndefined()
    expect(SONILO_T2M.outputKind).toBe('audio')
    expect(SONILO_T2M.provider).toBe('fal')
  })
})
