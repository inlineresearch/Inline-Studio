import { describe, expect, it } from 'vitest'
import {
  BUILD_STEP_CHOICES,
  SECONDS_PER_STEP,
  estimateMinutes,
  formatEstimate,
} from './characterTraining'

describe('character training estimates', () => {
  it('offers only step counts that were actually measured', () => {
    // Core refuses anything else (BUILD_STEP_CHOICES in studio/characters.py), so a value added
    // here without adding it there produces a build the engine rejects.
    expect([...BUILD_STEP_CHOICES]).toEqual([600, 1200, 2000])
  })

  it('reproduces the measured wall times it was derived from', () => {
    // FLUX.2: 9.7 / 19.4 / 32.3 min. Krea 2: 61.5 min at 600. Within a minute of the real runs.
    expect(estimateMinutes('flux2', 600)).toBeCloseTo(10, 0)
    expect(estimateMinutes('flux2', 1200)).toBeCloseTo(19, 0)
    expect(estimateMinutes('flux2', 2000)).toBeCloseTo(32, 0)
    expect(estimateMinutes('krea2', 600)).toBeCloseTo(62, 0)
  })

  it('has no estimate for an architecture that was never timed', () => {
    // Better a missing hint than an invented one: the rule is no duration without a measurement.
    expect(estimateMinutes('ltx25', 600)).toBeNull()
    expect(formatEstimate('ltx25', 600)).toBeNull()
    expect(SECONDS_PER_STEP.ltx25).toBeUndefined()
  })

  it('reads as hours once a build stops being a coffee break', () => {
    expect(formatEstimate('flux2', 600)).toBe('~10 min')
    expect(formatEstimate('krea2', 600)).toBe('~1 h 2 min')
    expect(formatEstimate('krea2', 2000)).toBe('~3 h 25 min')
  })
})
