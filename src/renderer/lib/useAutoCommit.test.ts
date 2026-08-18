import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { createAutoCommit } from './useAutoCommit'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

describe('createAutoCommit', () => {
  it('saves after the typing pause, not once per keystroke', () => {
    const commit = vi.fn()
    const auto = createAutoCommit(commit, 700)
    auto.schedule()
    auto.schedule()
    auto.schedule()
    expect(commit).not.toHaveBeenCalled()
    vi.advanceTimersByTime(700)
    expect(commit).toHaveBeenCalledTimes(1)
  })

  it('saves what is in the field now, not what was in it when the timer was armed', () => {
    // The whole point of the ref indirection in the hook: a stale closure would persist the text
    // from one keystroke ago, which is worse than not saving at all.
    let text = 'a'
    const seen: string[] = []
    const auto = createAutoCommit(() => seen.push(text), 700)
    auto.schedule()
    text = 'ab'
    vi.advanceTimersByTime(700)
    expect(seen).toEqual(['ab'])
  })

  it('flushes on blur and cancels what was pending', () => {
    const commit = vi.fn()
    const auto = createAutoCommit(commit, 700)
    auto.schedule()
    auto.flush()
    expect(commit).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(700)
    expect(commit).toHaveBeenCalledTimes(1)
  })

  it('reports whether a save is owed, so page-hide only fires when there is one', () => {
    const auto = createAutoCommit(vi.fn(), 700)
    expect(auto.pending()).toBe(false)
    auto.schedule()
    expect(auto.pending()).toBe(true)
    vi.advanceTimersByTime(700)
    expect(auto.pending()).toBe(false)
  })

  it('drops a pending save when the field goes away', () => {
    // A node usually unmounts because it was deleted; writing to a deleted item only raises.
    const commit = vi.fn()
    const auto = createAutoCommit(commit, 700)
    auto.schedule()
    auto.cancel()
    vi.advanceTimersByTime(700)
    expect(commit).not.toHaveBeenCalled()
  })
})
