/**
 * Every run Core knows about: queued, running, and recently finished, across projects and both
 * tabs. Core is one process, so its queue outlives the open project - a render started in one
 * project keeps going (and keeps reporting here) after switching to another.
 *
 * `events:activityChanged` carries the whole live list, so it replaces state rather than patching.
 */
import { create } from 'zustand'
import type { ActivityRun } from '@shared/types'
import { studio } from '@/lib/studio'
import { ipcErrorMessage } from '../lib/ipcError'

interface ActivityState {
  /** Queued + running, oldest first. */
  live: ActivityRun[]
  /** Finished runs for the open project, newest first. */
  history: ActivityRun[]
  error: string | null

  load: () => Promise<void>
  loadHistory: (limit?: number) => Promise<void>
  cancel: (runId: string) => Promise<void>
  /** Cancel everything queued or running, whichever project or tab it belongs to. */
  cancelAll: () => Promise<void>
  clearHistory: () => Promise<void>
  applySnapshot: (runs: ActivityRun[]) => void
}

export const useActivityStore = create<ActivityState>((set, get) => ({
  live: [],
  history: [],
  error: null,

  load: async () => {
    try {
      const res = await studio().activity.list()
      if (res.ok) set({ live: res.value })
    } catch {
      // A backend that predates the channel simply has no activity to show.
    }
  },

  loadHistory: async (limit) => {
    try {
      const res = await studio().activity.history(limit)
      if (res.ok) set({ history: res.value })
    } catch {
      // Same: history is additive, never worth surfacing an error over.
    }
  },

  cancel: async (runId) => {
    // Drop it locally first so the row stops looking live during the round trip.
    set((s) => ({ live: s.live.filter((r) => r.runId !== runId) }))
    try {
      const res = await studio().activity.cancel(runId)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  cancelAll: async () => {
    const runs = get().live
    set({ live: [] })
    // One call per run rather than a blanket cancel: training and fal runs do not share the
    // generation cancel path, and the registry already knows how to route each id.
    for (const run of runs) {
      try {
        const res = await studio().activity.cancel(run.runId)
        if (!res.ok) set({ error: res.error })
      } catch (e) {
        set({ error: ipcErrorMessage(e) })
      }
    }
  },

  clearHistory: async () => {
    set({ history: [] })
    try {
      await studio().activity.clearHistory()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  applySnapshot: (runs) => {
    const previous = get().live
    set({ live: runs })
    // A run leaving the live list finished, so the open project's history is now stale.
    if (previous.some((r) => !runs.find((n) => n.runId === r.runId))) void get().loadHistory()
  },
}))

/**
 * Subscribe once for the whole app (see App.tsx), not per view: the runs outlive both the tab that
 * started them and the project they belong to.
 */
export function subscribeActivityEvents(): () => void {
  const store = useActivityStore.getState()
  void store.load()
  void store.loadHistory()
  return studio().events.onActivityChanged((e) => {
    useActivityStore.getState().applySnapshot(e.runs)
  })
}
