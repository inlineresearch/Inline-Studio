/**
 * First-run coaching state: whether the Run and Edit-params hints have been shown, and which node
 * they currently point at.
 *
 * The "seen" flag is per device and global, not per project: you learn where Run is once, and a
 * second project should not teach it again. Persisted the same way as the canvas preferences, since
 * this is a view preference rather than project data.
 */
import { create } from 'zustand'

import type { CanvasSurface } from '@shared/types'

const SEEN_KEY = 'inline-studio.onboarding.starterHints'

function loadSeen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1'
  } catch {
    // Private mode or a quota error. Replaying the hints each session beats crashing the canvas.
    return false
  }
}

export interface HintTarget {
  itemId: string
  surface: CanvasSurface
}

interface OnboardingState {
  starterHintsSeen: boolean
  /** The node the hints point at. Transient, never persisted. */
  hintTarget: HintTarget | null
  /** Point the hints at a node. A no-op once they have been seen. */
  armHints: (target: HintTarget) => void
  dismissHints: () => void
  /** Escape hatch for tests and a future "replay tips" action. */
  resetSeen: () => void
}

export const useOnboardingStore = create<OnboardingState>((set, get) => ({
  starterHintsSeen: loadSeen(),
  hintTarget: null,
  armHints: (hintTarget) => {
    if (get().starterHintsSeen) return
    // Marked seen on show rather than on dismiss, so reloading mid-hint does not replay it.
    try {
      localStorage.setItem(SEEN_KEY, '1')
    } catch {
      /* not persisting is survivable; the hint still shows this session */
    }
    set({ hintTarget, starterHintsSeen: true })
  },
  dismissHints: () => set({ hintTarget: null }),
  resetSeen: () => {
    try {
      localStorage.removeItem(SEEN_KEY)
    } catch {
      /* nothing to undo */
    }
    set({ starterHintsSeen: false, hintTarget: null })
  },
}))
