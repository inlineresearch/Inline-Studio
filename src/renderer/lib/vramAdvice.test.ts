import { describe, expect, it } from 'vitest'

import {
  pickRecommended,
  readVram,
  recommendedLabel,
  tierFor,
  type StarterKey,
  type VramReading,
} from './vramAdvice'
import type { GpuStat, SystemStatsEvent } from '@shared/types'

const GB = 1024 ** 3
const gpu = (name: string, totalGb: number, index = 0): GpuStat => ({
  index,
  name,
  utilization: 0,
  memoryUsed: 0,
  memoryTotal: totalGb * GB,
  temperature: null,
})
const stats = (gpus: GpuStat[]): SystemStatsEvent => ({ cpu: 0, ramUsed: 0, ramTotal: 0, gpus })
const known = (totalGb: number): VramReading => ({ state: 'known', totalGb, name: 'x' })

describe('readVram', () => {
  it('is pending until the first broadcast arrives', () => {
    // systemStats is null for ~1.5s after mount; cards must render anyway.
    expect(readVram(null)).toEqual({ state: 'pending' })
  })

  it('is unknown rather than "no GPU" when nothing can be read', () => {
    // Apple Silicon, AMD, or missing pynvml all report an empty list, and those machines do run.
    expect(readVram(stats([]))).toEqual({ state: 'unknown' })
  })

  it('tiers on the largest card, not the first', () => {
    const reading = readVram(stats([gpu('display', 4, 0), gpu('compute', 24, 1)]))
    expect(reading).toMatchObject({ state: 'known', name: 'compute' })
    expect(reading.state === 'known' && Math.round(reading.totalGb)).toBe(24)
  })
})

describe('tierFor', () => {
  it('falls back to a static requirement when the hardware is unreadable', () => {
    for (const state of ['pending', 'unknown'] as const) {
      const advice = tierFor('flux2', { state })
      expect(advice.tier).toBeNull()
      expect(advice.note).toMatch(/VRAM/)
    }
  })

  it.each([
    ['zimage', 24, 'best'],
    ['zimage', 16, 'good'],
    ['zimage', 10, 'ok'],
    ['zimage', 8, 'heavy'],
    ['flux2', 24, 'best'],
    ['flux2', 16, 'good'],
    ['flux2', 12, 'ok'],
    ['flux2', 8, 'heavy'],
    ['krea2', 32, 'best'],
    ['krea2', 24, 'good'],
    ['krea2', 16, 'ok'],
    ['krea2', 12, 'heavy'],
    ['training', 24, 'best'],
    ['training', 16, 'ok'],
    ['training', 8, 'heavy'],
  ] as [StarterKey, number, string][])('%s at %s GB is %s', (key, gbTotal, tier) => {
    expect(tierFor(key, known(gbTotal)).tier).toBe(tier)
  })

  it('does not demote a card that advertises 16 GB but reports slightly under', () => {
    // A "16 GB" card reports ~15.9 GiB. Rounding it down a tier would be visibly wrong.
    expect(tierFor('flux2', known(15.6)).tier).toBe('good')
    expect(tierFor('zimage', known(15.6)).tier).toBe('good')
    // A genuinely smaller card still drops.
    expect(tierFor('flux2', known(14)).tier).toBe('ok')
  })

  it('always tells the user it will still run, even at the lowest tier', () => {
    for (const key of ['zimage', 'flux2', 'krea2', 'training'] as StarterKey[]) {
      const advice = tierFor(key, known(4))
      expect(advice.tier).toBe('heavy')
      expect(advice.note.toLowerCase()).toMatch(/will run|try /)
    }
  })
})

describe('pickRecommended', () => {
  it('returns the neutral starting point when the hardware is unreadable', () => {
    expect(pickRecommended({ state: 'pending' })).toBe('zimage')
    expect(pickRecommended({ state: 'unknown' })).toBe('zimage')
    expect(recommendedLabel({ state: 'unknown' })).toBe('Best place to start')
  })

  it('breaks ties lightest-first, so it is deterministic', () => {
    // At 24 GB, Z-Image and FLUX.2 are both `best`; the lighter one wins.
    expect(pickRecommended(known(24))).toBe('zimage')
    expect(recommendedLabel(known(24))).toBe('Recommended for your GPU')
  })

  it('still names exactly one card on a small GPU', () => {
    expect(pickRecommended(known(6))).toBe('zimage')
  })
})
