/**
 * The Trainer tab's canvas state - the same moodboard item/connector model, scoped to the `trainer`
 * surface so it never mixes with the Studio board.
 *
 * Deliberately much smaller than `moodboardStore`: the training graph has no frames, layers, takes
 * or asset imports, so it only needs load / add / move / connect / delete. Node chrome is shared
 * (`NodeFrame`, `NodeBadge`) via `BoardActionsProvider`, so the nodes still read as one card family.
 */
import { create } from 'zustand'
import type { MoodboardConnector, MoodboardItem } from '@shared/types'
import type { MoodboardItemPatch } from '@shared/ipc'
import { studio } from '@/lib/studio'
import { ipcErrorMessage } from '../lib/ipcError'

const SURFACE = 'trainer' as const

/** Node kinds the Trainer canvas can add (maps 1:1 to the `moodboard:add*` channels). */
export type TrainerNodeKind = 'trainDataset' | 'caption' | 'trainer' | 'lossGraph' | 'resource'

interface TrainerBoardState {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
  loading: boolean
  error: string | null
  load: () => Promise<void>
  addNode: (kind: TrainerNodeKind, x: number, y: number) => Promise<MoodboardItem | null>
  updateItem: (id: string, patch: MoodboardItemPatch) => Promise<void>
  /** Merge into an item's `data` (dataset/run/hyperparam selections live there). */
  patchData: (id: string, data: Record<string, unknown>) => Promise<void>
  deleteItem: (id: string) => Promise<void>
  connect: (
    fromItemId: string,
    toItemId: string,
    sourceHandle?: string | null,
    targetHandle?: string | null,
  ) => Promise<void>
  disconnect: (connectorId: string) => Promise<void>
  setError: (error: string | null) => void
  /** Node whose settings sidebar is open in the right gutter (params live off the node face). */
  settingsItemId: string | null
  toggleSettings: (itemId: string) => void
}

function addFor(kind: TrainerNodeKind, x: number, y: number) {
  const m = studio().moodboard
  switch (kind) {
    case 'trainDataset':
      return m.addTrainDataset(x, y)
    case 'caption':
      return m.addCaption(x, y)
    case 'trainer':
      return m.addTrainer(x, y)
    case 'lossGraph':
      return m.addLossGraph(x, y)
    case 'resource':
      return m.addResource(x, y, SURFACE)
  }
}

export const useTrainerBoardStore = create<TrainerBoardState>((set, get) => ({
  items: [],
  connectors: [],
  loading: false,
  error: null,

  load: async () => {
    set({ loading: true })
    try {
      const res = await studio().moodboard.list(SURFACE)
      if (!res.ok) return set({ error: res.error, loading: false })
      set({ items: res.value.items, connectors: res.value.connectors, loading: false })
    } catch (e) {
      set({ error: ipcErrorMessage(e), loading: false })
    }
  },

  addNode: async (kind, x, y) => {
    try {
      const res = await addFor(kind, x, y)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      set((s) => ({ items: [...s.items, res.value] }))
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  updateItem: async (id, patch) => {
    // Optimistic so dragging stays snappy, then persist.
    set((s) => ({
      items: s.items.map((it) =>
        it.id === id
          ? {
              ...it,
              x: patch.x ?? it.x,
              y: patch.y ?? it.y,
              width: patch.width ?? it.width,
              height: patch.height ?? it.height,
              data: patch.data ?? it.data,
            }
          : it,
      ),
    }))
    try {
      const res = await studio().moodboard.updateItem(id, patch)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  patchData: async (id, data) => {
    const item = get().items.find((it) => it.id === id)
    if (!item) return
    await get().updateItem(id, { data: { ...item.data, ...data } })
  },

  deleteItem: async (id) => {
    set((s) => ({
      items: s.items.filter((it) => it.id !== id),
      connectors: s.connectors.filter((c) => c.fromItemId !== id && c.toItemId !== id),
    }))
    try {
      const res = await studio().moodboard.deleteItem(id)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  connect: async (fromItemId, toItemId, sourceHandle = null, targetHandle = null) => {
    try {
      const res = await studio().moodboard.createConnector(
        fromItemId,
        toItemId,
        sourceHandle,
        targetHandle,
      )
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ connectors: [...s.connectors, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  disconnect: async (connectorId) => {
    set((s) => ({ connectors: s.connectors.filter((c) => c.id !== connectorId) }))
    try {
      const res = await studio().moodboard.deleteConnector(connectorId)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setError: (error) => set({ error }),

  settingsItemId: null,
  toggleSettings: (itemId) =>
    set((s) => ({ settingsItemId: s.settingsItemId === itemId ? null : itemId })),
}))
