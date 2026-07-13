/**
 * Director-node timeline state. The timeline is *derived* in main from the connections
 * wired into a director node (video inputs → video layer + L1 audio; audio inputs → L2);
 * this store just caches the resolved model per node and drives volume + render actions.
 */
import { create } from 'zustand'
import type { DirectorTimeline } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface TimelineState {
  /** Resolved (display) timeline per director item id. */
  timelineByOwner: Record<string, DirectorTimeline>
  /** Render progress (0..1) per director item id, or null when idle. */
  progressByOwner: Record<string, number | null>
  error: string | null

  /** Recompute a director's derived timeline from its connections. */
  resolve: (ownerItemId: string) => Promise<void>
  /** Persist the L1/L2 layer volumes (optimistic). */
  setVolumes: (ownerItemId: string, l1Volume: number, l2Volume: number) => Promise<void>
  setProgress: (ownerItemId: string, fraction: number | null) => void
  /** Render the low-res proxy preview; resolves true on success. */
  buildPreview: (ownerItemId: string) => Promise<boolean>
  /** Export the timeline to a user-chosen MP4; resolves the path written (or null). */
  exportTimeline: (ownerItemId: string) => Promise<string | null>
}

export const useTimelineStore = create<TimelineState>((set, get) => ({
  timelineByOwner: {},
  progressByOwner: {},
  error: null,

  resolve: async (ownerItemId) => {
    try {
      const res = await studio().timeline.resolve(ownerItemId)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ timelineByOwner: { ...s.timelineByOwner, [ownerItemId]: res.value } }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setVolumes: async (ownerItemId, l1Volume, l2Volume) => {
    // Optimistic: update the cached timeline so sliders + the rebuild trigger react now.
    set((s) => {
      const t = s.timelineByOwner[ownerItemId]
      if (!t) return {}
      return {
        timelineByOwner: { ...s.timelineByOwner, [ownerItemId]: { ...t, l1Volume, l2Volume } },
      }
    })
    try {
      const res = await studio().timeline.setVolumes(ownerItemId, l1Volume, l2Volume)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setProgress: (ownerItemId, fraction) =>
    set((s) => ({ progressByOwner: { ...s.progressByOwner, [ownerItemId]: fraction } })),

  buildPreview: async (ownerItemId) => {
    get().setProgress(ownerItemId, 0)
    try {
      const res = await studio().timeline.buildPreview(ownerItemId)
      if (!res.ok) {
        set({ error: res.error })
        return false
      }
      return res.value !== null
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return false
    } finally {
      get().setProgress(ownerItemId, null)
    }
  },

  exportTimeline: async (ownerItemId) => {
    get().setProgress(ownerItemId, 0)
    try {
      const res = await studio().timeline.export(ownerItemId)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    } finally {
      get().setProgress(ownerItemId, null)
    }
  },
}))
