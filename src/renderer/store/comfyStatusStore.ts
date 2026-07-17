/**
 * App-wide "is ComfyUI reachable?" state. The home screen reads it to show a connected /
 * not-connected status and to let the user set + test the address without leaving the page.
 *
 * One ref-counted poll loop runs while anything is subscribed (4s interval, hitting
 * `comfy:status` → `ping()`). A single slow/missed ping must NOT immediately flip to
 * "down": ComfyUI's single-threaded server routinely stalls past the ping timeout while
 * generating. So we only declare it down after 3 consecutive failures (~12s).
 */
import { create } from 'zustand'
import { studio } from '@/lib/studio'

/** Poll interval (ms). */
const POLL_MS = 4000
/** Consecutive failed pings before we believe ComfyUI is really down. */
const FAILURE_THRESHOLD = 3

interface ComfyStatusState {
  /** true = reachable, false = down, null = not yet checked. */
  running: boolean | null
  /** The ComfyUI URL the last check used (echoed back from main). */
  url: string
  /** A check is currently in flight. */
  checking: boolean
  /**
   * Begin polling (ref-counted). Returns an unsubscribe; the loop stops when the last
   * subscriber leaves. Safe to call from multiple components.
   */
  start: () => () => void
  /** Run one immediate, non-debounced check (for the "Test connection" button). */
  test: () => Promise<boolean>
}

let subscribers = 0
let timer: ReturnType<typeof setInterval> | null = null
let failures = 0

export const useComfyStatusStore = create<ComfyStatusState>((set, get) => {
  /** One ping; `immediate` skips the failure debounce (used by the Test button). */
  const runCheck = async (immediate: boolean): Promise<boolean> => {
    set({ checking: true })
    let down = false
    let nextUrl = get().url
    try {
      const res = await studio().comfy.status()
      if (res.ok) {
        nextUrl = res.value.url
        down = !res.value.running
      } else {
        down = true
      }
    } catch {
      down = true
    }

    if (!down) {
      failures = 0
      set({ running: true, url: nextUrl, checking: false })
      return true
    }
    if (immediate) {
      failures = FAILURE_THRESHOLD
      set({ running: false, url: nextUrl, checking: false })
      return false
    }
    // Debounced: only believe "down" after several consecutive misses.
    failures += 1
    if (failures >= FAILURE_THRESHOLD) set({ running: false, url: nextUrl, checking: false })
    else set({ url: nextUrl, checking: false })
    return false
  }

  return {
    running: null,
    url: '',
    checking: false,

    start: () => {
      subscribers += 1
      if (timer === null) {
        void runCheck(false)
        timer = setInterval(() => void runCheck(false), POLL_MS)
      }
      return () => {
        subscribers = Math.max(0, subscribers - 1)
        if (subscribers === 0 && timer !== null) {
          clearInterval(timer)
          timer = null
        }
      }
    },

    test: () => runCheck(true),
  }
})
