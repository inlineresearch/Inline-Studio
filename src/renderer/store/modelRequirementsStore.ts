/**
 * A Core node's model requirements + explicit-download state - the data behind the node's "missing
 * models" hint and popup. Nothing is auto-downloaded: a component is present only when the user
 * dropped files under models/ or downloaded it here (which writes into models/). Requirements are
 * per node *type* (every Z-Image node shares them), so a download benefits all nodes of that type.
 *
 * `load` fetches `models:requirements`; `download` kicks off `models:download` (fire-and-forget) and
 * progress arrives on the `events:modelDownload*` channels, wired in MoodboardPanel to the appliers.
 */
import { create } from 'zustand'
import type { ModelRequirements } from '@shared/coreNodes'
import type {
  ModelDownloadDoneEvent,
  ModelDownloadErrorEvent,
  ModelDownloadProgressEvent,
} from '@shared/types'
import { studio } from '@/lib/studio'
import { useCoreNodesStore } from './coreNodesStore'

/** Live state of one component's download (a value in `downloads[nodeType][componentId]`). */
export interface ComponentDownload {
  fraction: number
  status: string
  error?: string
}

interface ModelRequirementsState {
  /** Requirements per node type, from the last `load`. */
  byType: Record<string, ModelRequirements>
  /** The node type whose popup is open, or null. */
  openFor: string | null
  /** Per node type → per component id → live download state. */
  downloads: Record<string, Record<string, ComponentDownload>>
  load: (nodeType: string) => Promise<void>
  checkOnUse: (nodeType: string, reason: string) => Promise<number>
  open: (nodeType: string) => void
  close: () => void
  download: (nodeType: string, componentId: string) => Promise<void>
  onProgress: (e: ModelDownloadProgressEvent) => void
  onDone: (e: ModelDownloadDoneEvent) => void
  onError: (e: ModelDownloadErrorEvent) => void
}

export const useModelRequirementsStore = create<ModelRequirementsState>((set, get) => ({
  byType: {},
  openFor: null,
  downloads: {},

  load: async (nodeType) => {
    const res = await studio().models.requirements(nodeType)
    if (res.ok) set((s) => ({ byType: { ...s.byType, [nodeType]: res.value } }))
  },

  /** Load, then offer the registry whatever this node is missing. Used where a node is chosen. */
  checkOnUse: async (nodeType, reason) => {
    await get().load(nodeType)
    const reqs = get().byType[nodeType]
    if (!reqs || reqs.allPresent) return 0
    const { checkComponentModels } = await import('../lib/checkModels')
    return checkComponentModels(reqs.components, reason)
  },

  open: (nodeType) => set({ openFor: nodeType }),
  close: () => set({ openFor: null }),

  download: async (nodeType, componentId) => {
    const reqs = get().byType[nodeType]
    // Which component ids this call starts (a specific one, or every missing one for "all").
    const ids =
      componentId === 'all'
        ? (reqs?.components.filter((c) => !c.present).map((c) => c.id) ?? [])
        : [componentId]
    set((s) => {
      const forType = { ...(s.downloads[nodeType] ?? {}) }
      for (const id of ids) forType[id] = { fraction: 0, status: 'Starting…' }
      return { downloads: { ...s.downloads, [nodeType]: forType } }
    })
    await studio().models.download(nodeType, componentId)
  },

  onProgress: (e) =>
    set((s) => ({
      downloads: {
        ...s.downloads,
        [e.nodeType]: {
          ...(s.downloads[e.nodeType] ?? {}),
          [e.componentId]: { fraction: e.fraction, status: e.status ?? 'Downloading…' },
        },
      },
    })),

  onDone: (e) => {
    set((s) => {
      const forType = { ...(s.downloads[e.nodeType] ?? {}) }
      delete forType[e.componentId]
      return { downloads: { ...s.downloads, [e.nodeType]: forType } }
    })
    // The files are under models/ now - refresh requirements + the node palette (options/version).
    void get().load(e.nodeType)
    void useCoreNodesStore.getState().load()
  },

  onError: (e) =>
    set((s) => ({
      downloads: {
        ...s.downloads,
        [e.nodeType]: {
          ...(s.downloads[e.nodeType] ?? {}),
          [e.componentId]: {
            fraction: s.downloads[e.nodeType]?.[e.componentId]?.fraction ?? 0,
            status: 'Failed',
            error: e.error,
          },
        },
      },
    })),
}))

/** An in-progress (no-error) download for a node type, for the node's status pill; else null. */
export function activeDownload(
  downloads: Record<string, ComponentDownload>,
  reqs: ModelRequirements | undefined,
): { label: string; fraction: number; status: string } | null {
  for (const [id, dl] of Object.entries(downloads)) {
    if (dl.error) continue
    const label = reqs?.components.find((c) => c.id === id)?.label ?? 'model'
    return { label, fraction: dl.fraction, status: dl.status }
  }
  return null
}
