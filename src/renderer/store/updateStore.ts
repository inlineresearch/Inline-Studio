/** Auto-update state, fed by the main process's `events:update*` broadcasts. */
import { create } from 'zustand'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

const RELEASES_URL = 'https://github.com/inlineresearch/Inline-Studio/releases/latest'

type UpdateStatus = 'idle' | 'available' | 'downloading' | 'ready'

interface UpdateState {
  status: UpdateStatus
  version: string | null
  /** The running app version (from package.json), for display. */
  currentVersion: string | null
  /** 0–100 while downloading. */
  percent: number
  /** macOS: detect-only, so the banner links out instead of self-installing. */
  notifyOnly: boolean
  error: string | null
  /** Wire the main→renderer events; returns an unsubscribe fn. Call once on mount. */
  subscribeToEvents: () => () => void
  /** Fetch the running app version into `currentVersion`. */
  loadCurrentVersion: () => Promise<void>
  /** Quit and install the downloaded update (Windows/Linux). */
  install: () => Promise<void>
  /** Open the GitHub releases page (macOS notify-only path). */
  openReleases: () => Promise<void>
}

export const useUpdateStore = create<UpdateState>((set) => ({
  status: 'idle',
  version: null,
  currentVersion: null,
  percent: 0,
  notifyOnly: false,
  error: null,

  loadCurrentVersion: async () => {
    try {
      const res = await studio().app.version()
      if (res.ok) set({ currentVersion: res.value })
    } catch {
      // best-effort - the footer just omits the version if this fails
    }
  },

  subscribeToEvents: () => {
    const { events } = studio()
    const unsubs = [
      events.onUpdateAvailable((e) =>
        set({
          // notify-only updates have nothing more to do but surface the version.
          status: e.notifyOnly ? 'available' : 'downloading',
          version: e.version,
          notifyOnly: e.notifyOnly,
        }),
      ),
      events.onUpdateProgress((e) => set({ status: 'downloading', percent: e.percent })),
      events.onUpdateDownloaded((e) => set({ status: 'ready', version: e.version, percent: 100 })),
    ]
    return () => unsubs.forEach((u) => u())
  },

  install: async () => {
    try {
      const res = await studio().updates.quitAndInstall()
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  openReleases: async () => {
    try {
      await studio().shell.openExternal(RELEASES_URL)
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },
}))
