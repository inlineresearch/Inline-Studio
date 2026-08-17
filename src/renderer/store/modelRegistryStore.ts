/**
 * The published model list, and which files something asked for are absent.
 *
 * A missing file is reported whether or not the registry can supply it: the name and the folder are
 * what let someone place their own file, and only a match adds a download.
 */
import { create } from 'zustand'
import type { MissingModel, RegistryModel } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

export interface ModelRequest {
  filename: string
  category?: string
}

interface ModelRegistryState {
  entries: RegistryModel[]
  /** True when the registry was unreachable and these came from the on-disk cache. */
  stale: boolean
  loading: boolean
  error: string | null
  /** The open popup's missing files, or null when it is closed. */
  missing: MissingModel[] | null
  /** What opened the popup, so the user knows why they are seeing it. */
  reason: string
  downloading: Record<string, { fraction: number; status: string }>

  load: (refresh?: boolean) => Promise<void>
  check: (wanted: ModelRequest[], reason: string) => Promise<number>
  dismiss: () => void
  download: (modelId: string) => Promise<void>
}

export const useModelRegistryStore = create<ModelRegistryState>((set) => ({
  entries: [],
  stale: false,
  loading: false,
  error: null,
  missing: null,
  reason: '',
  downloading: {},

  load: async (refresh = false) => {
    set({ loading: true, error: null })
    try {
      const res = await studio().models.registry(refresh)
      if (!res.ok) return set({ loading: false, error: res.error })
      set({ entries: res.value.entries, stale: res.value.stale, loading: false })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  check: async (wanted, reason) => {
    if (wanted.length === 0) return 0
    try {
      const res = await studio().models.resolveMissing(wanted)
      if (!res.ok) return 0
      const missing = res.value.missing
      if (missing.length > 0) set({ missing, reason })
      return missing.length
    } catch {
      // A registry that cannot be reached must never block placing a node.
      return 0
    }
  },

  dismiss: () => set({ missing: null, reason: '' }),

  download: async (modelId) => {
    set((s) => ({
      downloading: { ...s.downloading, [modelId]: { fraction: 0, status: 'Starting…' } },
    }))
    try {
      const res = await studio().models.downloadRegistry(modelId)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },
}))

function without<T>(map: Record<string, T>, key: string): Record<string, T> {
  const next = { ...map }
  delete next[key]
  return next
}

/** Model downloads report against the node type they were started for; registry ones use this. */
const REGISTRY_NODE = 'registry'

export function subscribeModelRegistry(): () => void {
  const events = studio().events
  const onProgress = events.onModelDownloadProgress((e) => {
    if (e.nodeType !== REGISTRY_NODE) return
    useModelRegistryStore.setState((s) => ({
      downloading: {
        ...s.downloading,
        [e.componentId]: { fraction: e.fraction, status: e.status ?? '' },
      },
    }))
  })
  const onDone = events.onModelDownloadDone((e) => {
    if (e.nodeType !== REGISTRY_NODE) return
    useModelRegistryStore.setState((s) => ({ downloading: without(s.downloading, e.componentId) }))
    void useModelRegistryStore.getState().load()
  })
  const onError = events.onModelDownloadError((e) => {
    if (e.nodeType !== REGISTRY_NODE) return
    useModelRegistryStore.setState((s) => ({
      downloading: without(s.downloading, e.componentId),
      error: e.error,
    }))
  })
  return () => {
    onProgress()
    onDone()
    onError()
  }
}
