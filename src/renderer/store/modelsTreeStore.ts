/**
 * The read-only listing of every models root on disk, for the Models side panel.
 *
 * Refetched off `coreNodesStore.registryVersion`: dropping a weight file already bumps that (the
 * catalog fingerprint feeds it), so the panel refreshes without polling the filesystem itself.
 */
import { create } from 'zustand'
import type { ModelTreeRoot } from '@shared/types'
import { studio } from '@/lib/studio'
import { ipcErrorMessage } from '../lib/ipcError'

interface ModelsTreeState {
  roots: ModelTreeRoot[]
  loading: boolean
  error: string | null
  /** The registryVersion the current tree was fetched for, so a reload can be skipped. */
  loadedFor: string | null
  load: (registryVersion?: string) => Promise<void>
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
}))
