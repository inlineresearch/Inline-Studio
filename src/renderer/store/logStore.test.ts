import { beforeEach, describe, expect, it } from 'vitest'
import { useLogStore } from './logStore'

describe('the log buffer', () => {
  beforeEach(() => useLogStore.setState({ linesByRun: {} }))

  it('keeps one entry per line, so a multi-line write is still scrollable', () => {
    // The sender owns its formatting; a block written in one call would otherwise render as a
    // single unscrollable row, since the view never wraps.
    useLogStore.getState().append('r1', 'step 1\nstep 2\nstep 3')
    expect(useLogStore.getState().linesByRun.r1).toEqual(['step 1', 'step 2', 'step 3'])
  })

  it('keeps runs apart', () => {
    useLogStore.getState().append('r1', 'a')
    useLogStore.getState().append('r2', 'b')
    expect(useLogStore.getState().linesByRun).toEqual({ r1: ['a'], r2: ['b'] })
  })

  it('caps the buffer and keeps the newest', () => {
    // A long run must not grow it without bound; the tail is what the node shows.
    for (let i = 0; i < 3200; i++) useLogStore.getState().append('r1', `line ${i}`)
    const lines = useLogStore.getState().linesByRun.r1
    expect(lines.length).toBe(3000)
    expect(lines[lines.length - 1]).toBe('line 3199')
    expect(lines[0]).toBe('line 200')
  })

  it('clears one run without touching another', () => {
    useLogStore.getState().append('r1', 'a')
    useLogStore.getState().append('r2', 'b')
    useLogStore.getState().clear('r1')
    expect(useLogStore.getState().linesByRun).toEqual({ r2: ['b'] })
  })
})
