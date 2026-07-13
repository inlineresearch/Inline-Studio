/**
 * Moodboard state: the board's items + connectors. The canvas (React Flow) owns
 * transient drag positions; this store is the persisted source of truth and is
 * updated on discrete events (drag stop, resize end, text edit), each persisted
 * to main via window.inlineStudio.moodboard.
 */
import { create } from 'zustand'
import type { MoodboardItem, MoodboardConnector } from '@shared/types'
import type { MoodboardItemPatch } from '@shared/ipc'
import { ipcErrorMessage } from '../lib/ipcError'
import { useFrameStore } from './frameStore'

/** A board snapshot for the undo/redo stacks. */
interface BoardSnapshot {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
}

interface MoodboardState {
  items: MoodboardItem[]
  connectors: MoodboardConnector[]
  loading: boolean
  error: string | null
  /** Undo/redo history of board snapshots (most recent last). */
  past: BoardSnapshot[]
  future: BoardSnapshot[]

  load: () => Promise<void>
  /** Snapshot the current board onto the undo stack (clears redo). Call before a change. */
  record: () => void
  undo: () => Promise<void>
  redo: () => Promise<void>
  addAssetAt: (assetId: string, x: number, y: number) => Promise<void>
  addTextAt: (x: number, y: number) => Promise<void>
  addFrameFromAsset: (assetId: string, x: number, y: number) => Promise<void>
  addFrameItem: (frameId: string, x: number, y: number) => Promise<void>
  /** Place an existing frame node, parented to a layer when given. */
  addFrameItemInLayer: (
    frameId: string,
    x: number,
    y: number,
    parentId: string | null,
  ) => Promise<void>
  /** Create an empty frame and place its node on the canvas. Returns the new item. */
  addEmptyFrame: (x: number, y: number) => Promise<MoodboardItem | null>
  /** Add a Preview node. Returns the new item (for connection-drop suggestions). */
  addPreview: (x: number, y: number) => Promise<MoodboardItem | null>
  addLayer: (x: number, y: number) => Promise<void>
  addDirector: (x: number, y: number) => Promise<MoodboardItem | null>
  addTrim: (x: number, y: number) => Promise<MoodboardItem | null>
  /** Create a fal generation node for `modelId` and place it on the canvas. Returns the new item. */
  addGenNode: (modelId: string, x: number, y: number) => Promise<MoodboardItem | null>
  /** Create a text-prompt node (feeds a Generate node's prompt input). Returns the new item. */
  addPrompt: (x: number, y: number) => Promise<MoodboardItem | null>
  addCoreNode: (coreType: string, x: number, y: number) => Promise<MoodboardItem | null>
  /** Place an existing asset on the board, parented to a layer when given. */
  addFrameFromAssetInLayer: (
    assetId: string,
    x: number,
    y: number,
    parentId: string | null,
  ) => Promise<void>
  importAndPlace: (x: number, y: number) => Promise<MoodboardItem[]>
  /**
   * Duplicate a set of items (Figma/Miro copy-paste) shifted by `offset`. Frames
   * are cloned (new slot + inputs + workflow); selected layers carry their children
   * along. Returns the newly created items.
   */
  duplicateItems: (
    sources: MoodboardItem[],
    offset: { x: number; y: number },
  ) => Promise<MoodboardItem[]>
  /** `recordHistory: false` skips the undo snapshot — used by programmatic layout fits. */
  updateItem: (id: string, patch: MoodboardItemPatch, recordHistory?: boolean) => Promise<void>
  deleteItem: (id: string) => Promise<void>
  connect: (
    fromItemId: string,
    toItemId: string,
    sourceHandle?: string | null,
    targetHandle?: string | null,
  ) => Promise<void>
  disconnect: (connectorId: string) => Promise<void>
  setConnectorVolume: (connectorId: string, volume: number) => Promise<void>
  reset: () => void
}

function applyPatch(item: MoodboardItem, patch: MoodboardItemPatch): MoodboardItem {
  return {
    ...item,
    x: patch.x ?? item.x,
    y: patch.y ?? item.y,
    width: patch.width ?? item.width,
    height: patch.height ?? item.height,
    rotation: patch.rotation ?? item.rotation,
    zIndex: patch.zIndex ?? item.zIndex,
    data: patch.data ?? item.data,
    // parentId can be set to null (detach), so distinguish "absent" from "null".
    parentId: patch.parentId !== undefined ? patch.parentId : item.parentId,
  }
}

/**
 * Create a duplicate of one item at (x, y) under `parentId`. Frames are cloned in
 * main (new slot + inputs + workflow); other types are recreated and then patched
 * to carry over size and type-specific data. Returns the new item or null.
 */
async function copyOne(
  item: MoodboardItem,
  x: number,
  y: number,
  parentId: string | null,
): Promise<MoodboardItem | null> {
  const m = window.inlineStudio.moodboard
  let res
  switch (item.type) {
    case 'frame': {
      if (!item.frameId) return null
      const cloned = await window.inlineStudio.frames.clone(item.frameId)
      if (!cloned.ok) return null
      res = await m.addFrameItem(cloned.value.id, x, y)
      break
    }
    case 'asset':
      if (!item.assetId) return null
      res = await m.addAsset(item.assetId, x, y)
      break
    case 'text':
      res = await m.addText(x, y)
      break
    case 'preview':
      res = await m.addPreview(x, y)
      break
    case 'layer':
      res = await m.addLayer(x, y)
      break
    case 'director':
      res = await m.addDirector(x, y)
      break
    case 'trim':
      res = await m.addTrim(x, y)
      break
    default:
      return null
  }
  if (!res.ok) {
    return null
  }
  // Carry over size + parent; copy data only where it holds styling/labels (text,
  // layer) — for frame/asset/preview the identity lives in their own column.
  const patch: MoodboardItemPatch = { width: item.width, height: item.height, parentId }
  if (item.type === 'text' || item.type === 'layer') patch.data = item.data
  const patched = await m.updateItem(res.value.id, patch)
  return patched.ok ? patched.value : res.value
}

const HISTORY_LIMIT = 50

export const useMoodboardStore = create<MoodboardState>((set, get) => ({
  items: [],
  connectors: [],
  loading: false,
  error: null,
  past: [],
  future: [],

  load: async () => {
    set({ loading: true, error: null })
    try {
      const res = await window.inlineStudio.moodboard.list()
      if (!res.ok) return set({ loading: false, error: res.error })
      // A fresh load is a new baseline — clear undo history.
      set({
        items: res.value.items,
        connectors: res.value.connectors,
        loading: false,
        past: [],
        future: [],
      })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  record: () =>
    set((s) => ({
      past: [...s.past, { items: s.items, connectors: s.connectors }].slice(-HISTORY_LIMIT),
      future: [],
    })),

  undo: async () => {
    const s = get()
    const prev = s.past[s.past.length - 1]
    if (!prev) return
    set({
      past: s.past.slice(0, -1),
      future: [...s.future, { items: s.items, connectors: s.connectors }],
      items: prev.items,
      connectors: prev.connectors,
    })
    try {
      const res = await window.inlineStudio.moodboard.replaceBoard(prev.items, prev.connectors)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  redo: async () => {
    const s = get()
    const next = s.future[s.future.length - 1]
    if (!next) return
    set({
      future: s.future.slice(0, -1),
      past: [...s.past, { items: s.items, connectors: s.connectors }],
      items: next.items,
      connectors: next.connectors,
    })
    try {
      const res = await window.inlineStudio.moodboard.replaceBoard(next.items, next.connectors)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addAssetAt: async (assetId, x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addAsset(assetId, x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addTextAt: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addText(x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addFrameFromAsset: async (assetId, x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addFrameFromAsset(assetId, x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addFrameItem: async (frameId, x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addFrameItem(frameId, x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addEmptyFrame: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addEmptyFrame(x, y)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      set((s) => ({ items: [...s.items, res.value] }))
      // The new frame exists in main — refresh the frame store so its node resolves.
      await useFrameStore.getState().load()
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  addFrameItemInLayer: async (frameId, x, y, parentId) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addFrameItem(frameId, x, y)
      if (!res.ok) return set({ error: res.error })
      let item = res.value
      if (parentId) {
        const patched = await window.inlineStudio.moodboard.updateItem(item.id, { parentId })
        if (patched.ok) item = patched.value
      }
      set((s) => ({ items: [...s.items, item] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addPreview: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addPreview(x, y)
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

  addLayer: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addLayer(x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addDirector: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addDirector(x, y)
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

  addTrim: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addTrim(x, y)
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

  addGenNode: async (modelId, x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addGenNode(modelId, x, y)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      set((s) => ({ items: [...s.items, res.value] }))
      // The backing fal frame was created in main — refresh the frame store so the node resolves.
      await useFrameStore.getState().load()
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  addCoreNode: async (coreType, x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addCoreNode(coreType, x, y)
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

  addPrompt: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addPrompt(x, y)
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

  addFrameFromAssetInLayer: async (assetId, x, y, parentId) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.addFrameFromAsset(assetId, x, y)
      if (!res.ok) return set({ error: res.error })
      let item = res.value
      if (parentId) {
        const patched = await window.inlineStudio.moodboard.updateItem(item.id, { parentId })
        if (patched.ok) item = patched.value
      }
      set((s) => ({ items: [...s.items, item] }))
      // The frame + its input row were created in main; refresh the frame store so
      // the new FrameNode shows its name, input asset, and (future) takes.
      await useFrameStore.getState().load()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  connect: async (fromItemId, toItemId, sourceHandle = null, targetHandle = null) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.createConnector(
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
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.deleteConnector(connectorId)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ connectors: s.connectors.filter((c) => c.id !== connectorId) }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setConnectorVolume: async (connectorId, volume) => {
    // Optimistic: update the connector so the director re-resolves + rebuilds.
    set((s) => ({
      connectors: s.connectors.map((c) =>
        c.id === connectorId ? { ...c, data: { ...c.data, volume } } : c,
      ),
    }))
    try {
      const res = await window.inlineStudio.moodboard.setConnectorVolume(connectorId, volume)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  importAndPlace: async (x, y) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.importAndPlace(x, y)
      if (!res.ok) {
        set({ error: res.error })
        return []
      }
      set((s) => ({ items: [...s.items, ...res.value] }))
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return []
    }
  },

  duplicateItems: async (sources, offset) => {
    try {
      get().record()
      const created: MoodboardItem[] = []
      // Copy layers first so children can be re-parented to the new layer ids.
      const layerMap = new Map<string, string>()
      for (const layer of sources.filter((s) => s.type === 'layer')) {
        const copy = await copyOne(layer, layer.x + offset.x, layer.y + offset.y, null)
        if (copy) {
          layerMap.set(layer.id, copy.id)
          created.push(copy)
        }
      }

      // Items to copy: the selected non-layers, plus every child of a copied layer
      // (so a group duplicates with its contents). Dedupe by id.
      const items = useMoodboardStore.getState().items
      const toCopy = new Map<string, MoodboardItem>()
      for (const s of sources) if (s.type !== 'layer') toCopy.set(s.id, s)
      for (const it of items) if (it.parentId && layerMap.has(it.parentId)) toCopy.set(it.id, it)

      let clonedFrame = false
      for (const it of toCopy.values()) {
        const parentCopied = it.parentId != null && layerMap.has(it.parentId)
        const newParentId = parentCopied
          ? (layerMap.get(it.parentId as string) ?? null)
          : it.parentId
        // A child of a copied layer keeps its relative position (the layer already
        // moved); anything else is shifted by the paste offset.
        const x = parentCopied ? it.x : it.x + offset.x
        const y = parentCopied ? it.y : it.y + offset.y
        const copy = await copyOne(it, x, y, newParentId)
        if (copy) {
          created.push(copy)
          if (it.type === 'frame') clonedFrame = true
        }
      }

      if (created.length) set((s) => ({ items: [...s.items, ...created] }))
      // Cloned frames are new entities in main — refresh so their nodes resolve.
      if (clonedFrame) await useFrameStore.getState().load()
      return created
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return []
    }
  },

  updateItem: async (id, patch, recordHistory = true) => {
    if (recordHistory) get().record()
    // Optimistic: keep the canvas snappy, then persist.
    set((s) => ({ items: s.items.map((it) => (it.id === id ? applyPatch(it, patch) : it)) }))
    try {
      const res = await window.inlineStudio.moodboard.updateItem(id, patch)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  deleteItem: async (id) => {
    try {
      get().record()
      const res = await window.inlineStudio.moodboard.deleteItem(id)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({
        items: s.items.filter((it) => it.id !== id),
        connectors: s.connectors.filter((c) => c.fromItemId !== id && c.toItemId !== id),
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  reset: () => set({ items: [], connectors: [], error: null, past: [], future: [] }),
}))
