/**
 * Persist an edit as it is typed, not only when the field loses focus.
 *
 * Everything on the canvas saves implicitly, but a text field that only wrote on blur lost whatever
 * was typed if the app was closed with the cursor still in it - the prompt you just wrote, gone on
 * restart. A save is scheduled on each keystroke and forced on blur.
 */
import { useEffect, useMemo, useRef } from 'react'

/** Long enough that a save is not sent per keystroke, short enough to beat a quick close. */
const IDLE_MS = 700

export interface AutoCommit {
  /** Called per keystroke: (re)arms the idle save. */
  schedule: () => void
  /** Called on blur: saves now and cancels anything pending. */
  flush: () => void
  /** True while a save is owed, which is the only time the page-hide handler has work. */
  pending: () => boolean
  /** Drop a pending save without running it. */
  cancel: () => void
}

/** The timer half, free of React so it can be tested directly. */
export function createAutoCommit(commit: () => void, delay: number = IDLE_MS): AutoCommit {
  let timer: ReturnType<typeof setTimeout> | null = null
  const cancel = (): void => {
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
  }
  return {
    cancel,
    pending: () => timer !== null,
    flush: () => {
      cancel()
      commit()
    },
    schedule: () => {
      cancel()
      timer = setTimeout(() => {
        timer = null
        commit()
      }, delay)
    },
  }
}

export function useAutoCommit(commit: () => void, delay: number = IDLE_MS): AutoCommit {
  // Through a ref, so a save that fires later writes what is in the field now, not what was in it
  // when the timer was armed.
  const latest = useRef(commit)
  latest.current = commit
  const auto = useMemo(() => createAutoCommit(() => latest.current(), delay), [delay])

  // Best-effort on the way out: an in-flight POST may not survive the unload, which is why the
  // idle save above is the real mechanism rather than this.
  useEffect(() => {
    const onHide = (): void => {
      if (auto.pending()) auto.flush()
    }
    window.addEventListener('pagehide', onHide)
    document.addEventListener('visibilitychange', onHide)
    return () => {
      window.removeEventListener('pagehide', onHide)
      document.removeEventListener('visibilitychange', onHide)
      // Dropped rather than flushed: the usual reason a node unmounts is that it was deleted, and
      // writing to a deleted item would only raise.
      auto.cancel()
    }
  }, [auto])

  return auto
}
