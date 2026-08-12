/**
 * The read-only listing of every models root on disk, for the Models side panel.
 *
 * Refetched off `coreNodesStore.registryVersion`, which moves whenever Core rescans. Note the
 * catalog does NOT notice a file dropped in by hand on its own: something has to call
 * `models:rescan`, which is what `refresh` and the focus listener below are for.
 */
import { create } from 'zustand'
import type { ModelTreeRoot } from '@shared/types'
import { studio } from '@/lib/studio'
import { ipcErrorMessage } from '../lib/ipcError'
import { useCoreNodesStore } from './coreNodesStore'

interface ModelsTreeState {
  roots: ModelTreeRoot[]
  loading: boolean
  error: string | null
  /** The registryVersion the current tree was fetched for, so a reload can be skipped. */
  loadedFor: string | null
  load: (registryVersion?: string) => Promise<void>
  /** Re-scan disk, then reload the tree and every model picker. */
  refresh: () => Promise<void>
}

export const useModelsTreeStore = create<ModelsTreeState>((set, get) => ({
  roots: [],
  loading: false,
  error: null,
  loadedFor: null,

  load: async (registryVersion) => {
    if (registryVersion && get().loadedFor === registryVersion && get().roots.length > 0) return
    set({ loading: true, error: null })
    try {
      const res = await studio().models.tree()
      if (res.ok) set({ roots: res.value, loadedFor: registryVersion ?? null })
      else set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    } finally {
      set({ loading: false })
    }
  },

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      // The catalog caches its scan, so the pickers keep serving the old list until this runs.
      // Core broadcasts `modelsChanged`, which reloads the descriptors for every open client.
      const rescan = await studio().models.rescan()
      if (!rescan.ok) set({ error: rescan.error })
      const res = await studio().models.tree()
      if (res.ok) set({ roots: res.value, loadedFor: null })
      else set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    } finally {
      set({ loading: false })
    }
  },
}))

/**
 * Keep the model pickers honest without polling: Core tells us when the installed set changes, and
 * a rescan is triggered when the window regains focus, which is when a user who just dropped a file
 * into `models/` comes back to the app.
 */
export function subscribeModelChanges(): () => void {
  const unsub = studio().events.onModelsChanged(() => {
    void useCoreNodesStore.getState().load()
    void useModelsTreeStore.getState().load()
  })

  let last = 0
  const onFocus = (): void => {
    // A scan walks every models root, so throttle it rather than paying on every alt-tab.
    const now = Date.now()
    if (now - last < 10_000) return
    last = now
    void studio().models.rescan()
  }
  window.addEventListener('focus', onFocus)
  return () => {
    unsub()
    window.removeEventListener('focus', onFocus)
  }
}
