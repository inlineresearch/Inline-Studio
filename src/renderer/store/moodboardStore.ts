/**
 * Moodboard state: the board's items + connectors. The canvas (React Flow) owns
 * transient drag positions; this store is the persisted source of truth and is
 * updated on discrete events (drag stop, resize end, text edit), each persisted
 * to main via studio().moodboard.
 */
import { create } from 'zustand'
import type { MoodboardItem, MoodboardConnector } from '@shared/types'
import type { MoodboardItemPatch } from '@shared/ipc'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'
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
  /** Create a standalone "Load Assets" loader (a `type:'loader'` item holding asset refs). */
  addLoader: (x: number, y: number) => Promise<MoodboardItem | null>
  /** Create a "Control Space" 3D pose-editor node (renders an OpenPose control map). */
  addControlSpace: (x: number, y: number) => Promise<MoodboardItem | null>
  /** Append library assets to a loader's ordered asset list (deduped). */
  addLoaderAssets: (itemId: string, assetIds: string[]) => Promise<void>
  /** Remove one asset from a loader. */
  removeLoaderAsset: (itemId: string, assetId: string) => Promise<void>
  /** Move an asset to the front of a loader's list (its hero, fed downstream). */
  setLoaderHero: (itemId: string, assetId: string) => Promise<void>
  /** Add a Preview node. Returns the new item (for connection-drop suggestions). */
  addPreview: (x: number, y: number) => Promise<MoodboardItem | null>
  /** Utility: a read-only host-telemetry node (no handles). */
  addResource: (x: number, y: number) => Promise<MoodboardItem | null>
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
  /** Old id -> new id from the last `duplicateItems`. The returned array is reordered (layers
   * first, plus their children), so index matching cannot recover the pairing. */
  lastDuplicateIdMap: Map<string, string>
  /** `recordHistory: false` skips the undo snapshot - used by programmatic layout fits. */
  updateItem: (id: string, patch: MoodboardItemPatch, recordHistory?: boolean) => Promise<void>
  /** Restore the text of the prompt node wired into `nodeId`'s `prompt` input (no-op if none). Used
   * when switching a gen node's take history so the shown image's prompt is restored non-destructively. */
  setConnectedPromptText: (nodeId: string, text: string) => Promise<void>
  deleteItem: (id: string) => Promise<void>
  /** Delete one render from a Core node's output history (and its file). */
  removeCoreOutput: (itemId: string, takeId: string) => Promise<void>
  connect: (
    fromItemId: string,
    toItemId: string,
    sourceHandle?: string | null,
    targetHandle?: string | null,
    /** `false` folds this into the caller's undo step (duplicating a graph is one action). */
    recordHistory?: boolean,
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
  const m = studio().moodboard
  let res
  switch (item.type) {
    case 'frame': {
      if (!item.frameId) return null
      const cloned = await studio().frames.clone(item.frameId)
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
    case 'core': {
      const coreType = String((item.data?.core as { type?: string } | undefined)?.type ?? '')
      if (!coreType) return null
      res = await m.addCoreNode(coreType, x, y)
      break
    }
    case 'prompt':
      res = await m.addPrompt(x, y)
      break
    case 'loader':
      res = await m.addLoader(x, y)
      break
    case 'controlSpace':
      res = await m.addControlSpace(x, y)
      break
    default:
      return null
  }
  if (!res.ok) {
    return null
  }
  // Carry over size + parent, plus the data that makes the copy behave like the original. For
  // frame/asset/preview the identity lives in its own column, so there is nothing to carry.
  const patch: MoodboardItemPatch = { width: item.width, height: item.height, parentId }
  const data = item.data ?? {}
  if (item.type === 'text' || item.type === 'layer') patch.data = item.data
  else if (item.type === 'core') {
    // Settings only: the copy is a fresh slot, so it must not claim the original's takes.
    const core = (data.core ?? {}) as { type?: string; params?: Record<string, unknown> }
    patch.data = {
      ...res.value.data,
      core: { type: String(core.type ?? ''), params: core.params ?? {} },
    }
  } else if (item.type === 'prompt') {
    patch.data = { ...res.value.data, promptText: data.promptText ?? '' }
  } else if (item.type === 'loader') {
    patch.data = { ...res.value.data, assetIds: data.assetIds ?? [] }
  } else if (item.type === 'controlSpace') {
    patch.data = {
      ...res.value.data,
      controlAssetId: data.controlAssetId,
      controlScene: data.controlScene,
    }
  }
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
  lastDuplicateIdMap: new Map(),

  load: async () => {
    set({ loading: true, error: null })
    try {
      const res = await studio().moodboard.list()
      if (!res.ok) return set({ loading: false, error: res.error })
      // A fresh load is a new baseline - clear undo history.
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
      const res = await studio().moodboard.replaceBoard(prev.items, prev.connectors)
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
      const res = await studio().moodboard.replaceBoard(next.items, next.connectors)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addAssetAt: async (assetId, x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addAsset(assetId, x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addTextAt: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addText(x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addFrameFromAsset: async (assetId, x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addFrameFromAsset(assetId, x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addFrameItem: async (frameId, x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addFrameItem(frameId, x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addEmptyFrame: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addEmptyFrame(x, y)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      set((s) => ({ items: [...s.items, res.value] }))
      // The new frame exists in main - refresh the frame store so its node resolves.
      await useFrameStore.getState().load()
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  addLoader: async (x, y) => {
    // A "Load Assets" node is a standalone `type:'loader'` item holding library asset refs in its
    // data (no frame). It renders as a resizable viewer and feeds its hero asset downstream.
    try {
      get().record()
      const res = await studio().moodboard.addLoader(x, y)
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

  addControlSpace: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addControlSpace(x, y)
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

  addLoaderAssets: async (itemId, assetIds) => {
    const item = get().items.find((i) => i.id === itemId)
    if (!item) return
    const current = item.data.assetIds ?? []
    const next = [...current, ...assetIds.filter((id) => !current.includes(id))]
    if (next.length === current.length) return
    await get().updateItem(itemId, { data: { ...item.data, assetIds: next } })
  },

  removeLoaderAsset: async (itemId, assetId) => {
    const item = get().items.find((i) => i.id === itemId)
    if (!item) return
    const next = (item.data.assetIds ?? []).filter((id) => id !== assetId)
    await get().updateItem(itemId, { data: { ...item.data, assetIds: next } })
  },

  removeCoreOutput: async (itemId, takeId) => {
    const item = get().items.find((i) => i.id === itemId)
    const core = item?.data.core
    if (!core) return
    // Optimistic: drop it from the history, promoting the newest remaining as active if needed.
    const outputs = (core.outputs ?? []).filter((o) => o.takeId !== takeId)
    const output = core.output?.takeId === takeId ? (outputs[0] ?? undefined) : core.output
    set((s) => ({
      items: s.items.map((it) =>
        it.id === itemId ? { ...it, data: { ...it.data, core: { ...core, output, outputs } } } : it,
      ),
    }))
    try {
      const res = await studio().moodboard.removeCoreOutput(itemId, takeId)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setLoaderHero: async (itemId, assetId) => {
    const item = get().items.find((i) => i.id === itemId)
    if (!item) return
    const rest = (item.data.assetIds ?? []).filter((id) => id !== assetId)
    await get().updateItem(itemId, { data: { ...item.data, assetIds: [assetId, ...rest] } })
  },

  addFrameItemInLayer: async (frameId, x, y, parentId) => {
    try {
      get().record()
      const res = await studio().moodboard.addFrameItem(frameId, x, y)
      if (!res.ok) return set({ error: res.error })
      let item = res.value
      if (parentId) {
        const patched = await studio().moodboard.updateItem(item.id, { parentId })
        if (patched.ok) item = patched.value
      }
      set((s) => ({ items: [...s.items, item] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addResource: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addResource(x, y)
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

  addPreview: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addPreview(x, y)
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
      const res = await studio().moodboard.addLayer(x, y)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({ items: [...s.items, res.value] }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addDirector: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.addDirector(x, y)
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
      const res = await studio().moodboard.addTrim(x, y)
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
      const res = await studio().moodboard.addGenNode(modelId, x, y)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      set((s) => ({ items: [...s.items, res.value] }))
      // The backing fal frame was created in main - refresh the frame store so the node resolves.
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
      const res = await studio().moodboard.addCoreNode(coreType, x, y)
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
      const res = await studio().moodboard.addPrompt(x, y)
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
      const res = await studio().moodboard.addFrameFromAsset(assetId, x, y)
      if (!res.ok) return set({ error: res.error })
      let item = res.value
      if (parentId) {
        const patched = await studio().moodboard.updateItem(item.id, { parentId })
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

  connect: async (
    fromItemId,
    toItemId,
    sourceHandle = null,
    targetHandle = null,
    recordHistory = true,
  ) => {
    try {
      if (recordHistory) get().record()
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
    try {
      get().record()
      const res = await studio().moodboard.deleteConnector(connectorId)
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
      const res = await studio().moodboard.setConnectorVolume(connectorId, volume)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  importAndPlace: async (x, y) => {
    try {
      get().record()
      const res = await studio().moodboard.importAndPlace(x, y)
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

      // Old id -> new id, for callers that must re-create the wiring between the copies.
      const idMap = new Map<string, string>(layerMap)
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
          idMap.set(it.id, copy.id)
          if (it.type === 'frame') clonedFrame = true
        }
      }

      if (created.length) set((s) => ({ items: [...s.items, ...created] }))
      // Cloned frames are new entities in main - refresh so their nodes resolve.
      if (clonedFrame) await useFrameStore.getState().load()
      set({ lastDuplicateIdMap: idMap })
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
      const res = await studio().moodboard.updateItem(id, patch)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setConnectedPromptText: async (nodeId, text) => {
    const { items, connectors } = get()
    const conn = connectors.find(
      (c) =>
        c.toItemId === nodeId && (c.data as { targetHandle?: string }).targetHandle === 'prompt',
    )
    if (!conn) return
    const promptNode = items.find((i) => i.id === conn.fromItemId && i.type === 'prompt')
    if (!promptNode || promptNode.data.promptText === text) return
    await get().updateItem(promptNode.id, { data: { ...promptNode.data, promptText: text } })
  },

  deleteItem: async (id) => {
    try {
      get().record()
      const res = await studio().moodboard.deleteItem(id)
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
