import { describe, expect, it } from 'vitest'
import type { InstallPhase } from '@shared/extensions'
import { stateOf } from './installStepsModel'

// Index into the STEPS array: 0 fetch, 1 validate, 2 scan, 3 preflight, 4 resolve, 5 lock,
// 6 register, 7 activate.
const FETCH = 0
const SCAN = 2
const RESOLVE = 4
const ACTIVATE = 7

describe('install stepper', () => {
  it('marks earlier phases done and the current one active', () => {
    const seen: InstallPhase[] = ['fetch', 'validate', 'scan']
    expect(stateOf(FETCH, 'scan', seen, false)).toBe('done')
    expect(stateOf(SCAN, 'scan', seen, false)).toBe('active')
    expect(stateOf(RESOLVE, 'scan', seen, false)).toBe('pending')
  })

  it('marks every step done once the run finishes', () => {
    // A fast local install can emit phases quicker than React renders, so completion must not
    // depend on having observed each one.
    expect(stateOf(FETCH, 'done', [], false)).toBe('done')
    expect(stateOf(ACTIVATE, 'done', [], false)).toBe('done')
  })

  it('marks the failing phase failed and leaves later ones pending', () => {
    const seen: InstallPhase[] = ['fetch', 'validate', 'scan']
    expect(stateOf(SCAN, 'scan', seen, true)).toBe('failed')
    expect(stateOf(FETCH, 'scan', seen, true)).toBe('done')
    expect(stateOf(ACTIVATE, 'scan', seen, true)).toBe('pending')
  })

  it('treats an unstarted install as all pending', () => {
    expect(stateOf(FETCH, 'idle', [], false)).toBe('pending')
    expect(stateOf(ACTIVATE, 'idle', [], false)).toBe('pending')
  })
})
