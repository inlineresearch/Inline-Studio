import { beforeEach, describe, expect, it } from 'vitest'
import { isCoreConnected, setCoreConnected, subscribeCoreConnection } from './connection'

beforeEach(() => {
  setCoreConnected(false)
})

describe('core connection state', () => {
  it('starts disconnected, so the dot is not green before the socket opens', () => {
    expect(isCoreConnected()).toBe(false)
  })

  it('notifies subscribers on a change', () => {
    const seen: boolean[] = []
    const stop = subscribeCoreConnection((c) => seen.push(c))
    setCoreConnected(true)
    setCoreConnected(false)
    stop()
    expect(seen).toEqual([true, false])
  })

  it('does not notify when the state is unchanged', () => {
    // useSyncExternalStore re-renders on every notify, and the reconnect loop can report the same
    // state repeatedly, so a repeat must be a no-op.
    const seen: boolean[] = []
    const stop = subscribeCoreConnection((c) => seen.push(c))
    setCoreConnected(true)
    setCoreConnected(true)
    stop()
    expect(seen).toEqual([true])
  })

  it('stops notifying after unsubscribe', () => {
    const seen: boolean[] = []
    const stop = subscribeCoreConnection((c) => seen.push(c))
    stop()
    setCoreConnected(true)
    expect(seen).toEqual([])
  })

  it('survives a subscriber unsubscribing during a notify', () => {
    // The set is copied before iterating; without that this drops the second listener.
    const seen: string[] = []
    const stopA = subscribeCoreConnection(() => {
      seen.push('a')
      stopA()
    })
    subscribeCoreConnection(() => seen.push('b'))
    setCoreConnected(true)
    expect(seen).toEqual(['a', 'b'])
  })
})
