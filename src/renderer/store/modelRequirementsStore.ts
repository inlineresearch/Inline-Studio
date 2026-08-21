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
  /** Requirements per cache key (usually the node type), from the last `load`. */
  byType: Record<string, ModelRequirements>
  /** What each key was asked for, so a reload after a download repeats the same question. */
  asked: Record<string, { nodeType: string; params?: Record<string, unknown> }>
  /** The node type whose popup is open, or null. */
  openFor: string | null
  /** Per node type → per component id → live download state. */
  downloads: Record<string, Record<string, ComponentDownload>>
  /**
   * `key` is where the answer is cached, defaulting to the node type.
   *
   * A node type is not always enough: Train LoRA's components follow the architecture in its own
   * settings, so two of them on different archs would otherwise overwrite each other's answer.
   */
  load: (nodeType: string, params?: Record<string, unknown>, key?: string) => Promise<void>
  checkOnUse: (
    nodeType: string,
    reason: string,
    params?: Record<string, unknown>,
    key?: string,
  ) => Promise<number>
  open: (key: string) => void
  close: () => void
  download: (
    key: string,
    componentId: string,
    nodeType?: string,
    params?: Record<string, unknown>,
  ) => Promise<void>
  onProgress: (e: ModelDownloadProgressEvent) => void
  onDone: (e: ModelDownloadDoneEvent) => void
  onError: (e: ModelDownloadErrorEvent) => void
}

/** Every cache key that asked about this node type: the type itself, plus any keyed entries. */
function keysFor(state: ModelRequirementsState, nodeType: string): string[] {
  const keys = Object.entries(state.asked)
    .filter(([, asked]) => asked.nodeType === nodeType)
    .map(([key]) => key)
  return keys.length ? keys : [nodeType]
}

function patch(
  downloads: Record<string, Record<string, ComponentDownload>>,
  keys: string[],
  edit: (forKey: Record<string, ComponentDownload>) => Record<string, ComponentDownload>,
): Record<string, Record<string, ComponentDownload>> {
  const next = { ...downloads }
  for (const key of keys) next[key] = edit(next[key] ?? {})
  return next
}

export const useModelRequirementsStore = create<ModelRequirementsState>((set, get) => ({
  byType: {},
  asked: {},
  openFor: null,
  downloads: {},

  load: async (nodeType, params, key) => {
    const at = key ?? nodeType
    const res = await studio().models.requirements(nodeType, params)
    if (!res.ok) return
    set((s) => ({
      byType: { ...s.byType, [at]: res.value },
      asked: { ...s.asked, [at]: { nodeType, params } },
    }))
  },

  /** Load, then offer the registry whatever this node is missing. Used where a node is chosen. */
  checkOnUse: async (nodeType, reason, params, key) => {
    await get().load(nodeType, params, key)
    const reqs = get().byType[key ?? nodeType]
    if (!reqs || reqs.allPresent) return 0
    const { checkComponentModels } = await import('../lib/checkModels')
    return checkComponentModels(reqs.components, reason)
  },

  open: (nodeType) => set({ openFor: nodeType }),
  close: () => set({ openFor: null }),

  download: async (key, componentId, nodeType, params) => {
    const reqs = get().byType[key]
    // Which component ids this call starts (a specific one, or every missing one for "all").
    const ids =
      componentId === 'all'
        ? (reqs?.components.filter((c) => !c.present).map((c) => c.id) ?? [])
        : [componentId]
    set((s) => {
      const forType = { ...(s.downloads[key] ?? {}) }
      for (const id of ids) forType[id] = { fraction: 0, status: 'Starting…' }
      return { downloads: { ...s.downloads, [key]: forType } }
    })
    await studio().models.download(nodeType ?? key, componentId, params)
  },

  onProgress: (e) =>
    set((s) => ({
      downloads: patch(s.downloads, keysFor(s, e.nodeType), (forKey) => ({
        ...forKey,
        [e.componentId]: { fraction: e.fraction, status: e.status ?? 'Downloading…' },
      })),
    })),

  onDone: (e) => {
    set((s) => ({
      downloads: patch(s.downloads, keysFor(s, e.nodeType), (forKey) => {
        const next = { ...forKey }
        delete next[e.componentId]
        return next
      }),
    }))
    // The files are under models/ now - refresh requirements + the node palette (options/version).
    // Re-asked exactly as before, because a keyed entry's answer depends on the params it carried.
    for (const key of keysFor(get(), e.nodeType)) {
      const asked = get().asked[key]
      if (asked) void get().load(asked.nodeType, asked.params, key)
    }
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
