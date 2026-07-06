import { describe, it, expect } from 'vitest'
import { progressFromLogs, resolveQueueUrls } from './client'

describe('progressFromLogs', () => {
  it('returns null with no logs', () => {
    expect(progressFromLogs(undefined)).toBeNull()
    expect(progressFromLogs([])).toBeNull()
  })

  it('reads "step N/M" into a step label + fraction', () => {
    const p = progressFromLogs([{ message: 'sampling 15/30' }])
    expect(p?.status).toBe('Step 15/30')
    // 0.1 + 0.85 * 0.5 = 0.525
    expect(p?.fraction).toBeCloseTo(0.525, 3)
  })

  it('reads "N of M"', () => {
    const p = progressFromLogs([{ message: 'step 30 of 30 done' }])
    expect(p?.status).toBe('Step 30/30')
    expect(p?.fraction).toBeCloseTo(0.95, 3)
  })

  it('reads a percentage when no step count is present', () => {
    const p = progressFromLogs([{ message: 'progress: 40%' }])
    expect(p?.status).toBe('40%')
    // 0.1 + 0.85 * 0.4 = 0.44
    expect(p?.fraction).toBeCloseTo(0.44, 3)
  })

  it('uses the most recent progress line', () => {
    const p = progressFromLogs([
      { message: 'starting 1/10' },
      { message: 'working 7/10' },
      { message: 'no numbers here' },
    ])
    expect(p?.status).toBe('Step 7/10')
  })

  it('ignores lines with no parseable progress', () => {
    expect(progressFromLogs([{ message: 'Loading model weights…' }])).toBeNull()
  })
})

describe('resolveQueueUrls', () => {
  it('prefers the URLs fal returns', () => {
    const h = resolveQueueUrls('openai/gpt-image-2/image-to-image', {
      request_id: 'r1',
      status_url: 'https://queue.fal.run/openai/gpt-image-2/requests/r1/status',
      response_url: 'https://queue.fal.run/openai/gpt-image-2/requests/r1',
    })
    expect(h.statusUrl).toBe('https://queue.fal.run/openai/gpt-image-2/requests/r1/status')
    expect(h.responseUrl).toBe('https://queue.fal.run/openai/gpt-image-2/requests/r1')
  })

  it('derives the result URL from the status URL when response_url is missing', () => {
    const h = resolveQueueUrls('openai/gpt-image-2/image-to-image', {
      request_id: 'r1',
      status_url: 'https://queue.fal.run/openai/gpt-image-2/requests/r1/status',
    })
    // No sub-path, and no trailing /status — this is the 404 the fallback used to cause.
    expect(h.responseUrl).toBe('https://queue.fal.run/openai/gpt-image-2/requests/r1')
  })

  it('falls back to the base app id (not the sub-path) when fal returns no URLs', () => {
    const h = resolveQueueUrls('openai/gpt-image-2/image-to-image', { request_id: 'r1' })
    expect(h.statusUrl).toBe('https://queue.fal.run/openai/gpt-image-2/requests/r1/status')
    expect(h.responseUrl).toBe('https://queue.fal.run/openai/gpt-image-2/requests/r1')
    expect(h.responseUrl).not.toContain('image-to-image')
  })

  it('leaves a plain (no sub-path) endpoint unchanged in the fallback', () => {
    const h = resolveQueueUrls('openai/gpt-image-2', { request_id: 'r2' })
    expect(h.statusUrl).toBe('https://queue.fal.run/openai/gpt-image-2/requests/r2/status')
  })
})
