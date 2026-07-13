/**
 * Frame timeline state: ordered frames, each with multiple inputs (library assets)
 * and multiple outputs (takes). The card shows compact stacks; the Frame Inspector
 * manages the full grids. Work happens in main via studio().frames / .comfy.
 */
import { create } from 'zustand'
import type { Frame, Take, FrameInput, ComfyOutput } from '@shared/types'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'

interface FrameState {
  frames: Frame[]
  /** Inputs (library assets) per frame id, in order. */
  inputsByFrame: Record<string, FrameInput[]>
  /** Takes (outputs) per frame id, newest first. */
  takesByFrame: Record<string, Take[]>
  selectedId: string | null
  loading: boolean
  /** Frame id currently mid-action (link/pull), for in-card spinners. */
  busyId: string | null
  error: string | null
  /** Transient status message (e.g. export summary). */
  notice: string | null

  load: () => Promise<void>
  importAsFrames: () => Promise<void>
  addFromAssets: (assetIds: string[]) => Promise<void>
  addInputs: (frameId: string, assetIds: string[]) => Promise<void>
  /** Link another frame's output as an input (resolves to its hero take). */
  addSourceInput: (frameId: string, sourceFrameId: string) => Promise<void>
  removeInput: (frameId: string, assetId: string) => Promise<void>
  /** Remove one input by its row id (works for asset AND flow-link inputs). */
  removeInputById: (frameId: string, inputId: string) => Promise<void>
  reorderInputs: (frameId: string, orderedAssetIds: string[]) => Promise<void>
  /** Resolve an `unset` chooser frame to an engine (comfy | fal); returns the updated frame. */
  setProvider: (
    frameId: string,
    provider: 'comfy' | 'fal',
    modelId?: string,
  ) => Promise<Frame | null>
  setHero: (frameId: string, takeId: string | null) => Promise<void>
  deleteTake: (takeId: string) => Promise<void>
  rename: (id: string, name: string) => Promise<void>
  reorder: (orderedIds: string[]) => Promise<void>
  remove: (id: string) => Promise<void>
  /** Duplicate a frame (inputs + workflow); returns the new frame. */
  clone: (id: string) => Promise<Frame | null>
  /** Detach the frame's ComfyUI workflow link. */
  unlink: (id: string) => Promise<void>
  linkFrame: (id: string) => Promise<Frame | null>
  uploadInputs: (id: string) => Promise<void>
  /** Pull the frame's workflow from ComfyUI into the durable project copy. */
  pullWorkflow: (id: string) => Promise<void>
  /** Persist the live (possibly unsaved) ComfyUI graph for a frame into the project. */
  saveLiveWorkflow: (id: string, workflow: unknown) => Promise<void>
  pullResult: (id: string) => Promise<void>
  captureOutput: (frameId: string, output: ComfyOutput) => Promise<void>
  exportFrames: () => Promise<void>
  select: (id: string | null) => void
  reset: () => void
}

function groupByFrame<T extends { frameId: string }>(items: T[]): Record<string, T[]> {
  const map: Record<string, T[]> = {}
  for (const item of items) (map[item.frameId] ??= []).push(item)
  return map
}

export const useFrameStore = create<FrameState>((set, get) => ({
  frames: [],
  inputsByFrame: {},
  takesByFrame: {},
  selectedId: null,
  loading: false,
  busyId: null,
  error: null,
  notice: null,

  load: async () => {
    set({ loading: true, error: null })
    try {
      const [framesRes, inputsRes, takesRes] = await Promise.all([
        studio().frames.list(),
        studio().frames.listInputs(),
        studio().frames.listAllTakes(),
      ])
      if (!framesRes.ok) return set({ loading: false, error: framesRes.error })
      if (!inputsRes.ok) return set({ loading: false, error: inputsRes.error })
      if (!takesRes.ok) return set({ loading: false, error: takesRes.error })
      set({
        frames: framesRes.value,
        inputsByFrame: groupByFrame(inputsRes.value),
        takesByFrame: groupByFrame(takesRes.value),
        loading: false,
      })
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  importAsFrames: async () => {
    set({ loading: true, error: null })
    try {
      const res = await studio().frames.importAsFrames()
      if (!res.ok) return set({ loading: false, error: res.error })
      await get().load() // refresh frames + their inputs
    } catch (e) {
      set({ loading: false, error: ipcErrorMessage(e) })
    }
  },

  addFromAssets: async (assetIds) => {
    try {
      for (const assetId of assetIds) {
        const res = await studio().frames.addFromAsset(assetId)
        if (!res.ok) return set({ error: res.error })
      }
      await get().load()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addInputs: async (frameId, assetIds) => {
    try {
      const res = await studio().frames.addInputs(frameId, assetIds)
      if (!res.ok) return set({ error: res.error })
      const added = res.value
      if (added.length === 0) return
      set((s) => {
        const existing = s.inputsByFrame[frameId] ?? []
        const ids = new Set(existing.map((i) => i.id))
        const merged = [...existing, ...added.filter((i) => !ids.has(i.id))]
        return { inputsByFrame: { ...s.inputsByFrame, [frameId]: merged } }
      })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  addSourceInput: async (frameId, sourceFrameId) => {
    try {
      const res = await studio().frames.addSourceInput(frameId, sourceFrameId)
      if (!res.ok) return set({ error: res.error })
      set((s) => {
        const existing = s.inputsByFrame[frameId] ?? []
        if (existing.some((i) => i.id === res.value.id)) return {}
        return { inputsByFrame: { ...s.inputsByFrame, [frameId]: [...existing, res.value] } }
      })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  removeInput: async (frameId, assetId) => {
    try {
      const res = await studio().frames.removeInput(frameId, assetId)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({
        inputsByFrame: {
          ...s.inputsByFrame,
          [frameId]: (s.inputsByFrame[frameId] ?? []).filter((i) => i.assetId !== assetId),
        },
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  removeInputById: async (frameId, inputId) => {
    try {
      const res = await studio().frames.removeInputById(frameId, inputId)
      if (!res.ok) return set({ error: res.error })
      set((s) => ({
        inputsByFrame: {
          ...s.inputsByFrame,
          [frameId]: (s.inputsByFrame[frameId] ?? []).filter((i) => i.id !== inputId),
        },
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setProvider: async (frameId, provider, modelId) => {
    try {
      const res = await studio().frames.setProvider(frameId, provider, modelId)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      const frame = res.value
      set((s) => ({ frames: s.frames.map((f) => (f.id === frameId ? frame : f)) }))
      return frame
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  reorderInputs: async (frameId, orderedAssetIds) => {
    set((s) => {
      const byAsset = new Map((s.inputsByFrame[frameId] ?? []).map((i) => [i.assetId, i]))
      const next = orderedAssetIds
        .map((assetId, position) => {
          const input = byAsset.get(assetId)
          return input ? { ...input, position } : null
        })
        .filter((x): x is FrameInput => x !== null)
      return { inputsByFrame: { ...s.inputsByFrame, [frameId]: next } }
    })
    try {
      const res = await studio().frames.reorderInputs(frameId, orderedAssetIds)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setHero: async (frameId, takeId) => {
    set((s) => ({
      frames: s.frames.map((sh) => (sh.id === frameId ? { ...sh, heroTakeId: takeId } : sh)),
    }))
    try {
      const res = await studio().frames.setHero(frameId, takeId)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  deleteTake: async (takeId) => {
    try {
      const res = await studio().frames.deleteTake(takeId)
      if (!res.ok) return set({ error: res.error })
      set((s) => {
        const takesByFrame: Record<string, Take[]> = {}
        for (const [frameId, takes] of Object.entries(s.takesByFrame)) {
          takesByFrame[frameId] = takes.filter((t) => t.id !== takeId)
        }
        return {
          takesByFrame,
          frames: s.frames.map((sh) =>
            sh.heroTakeId === takeId ? { ...sh, heroTakeId: null } : sh,
          ),
        }
      })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  linkFrame: async (id) => {
    set({ busyId: id, error: null })
    try {
      const res = await studio().comfy.linkFrame(id)
      if (!res.ok) {
        set({ error: res.error, busyId: null })
        return null
      }
      const frame = res.value
      set((s) => ({ frames: s.frames.map((sh) => (sh.id === id ? frame : sh)), busyId: null }))
      return frame
    } catch (e) {
      set({ error: ipcErrorMessage(e), busyId: null })
      return null
    }
  },

  uploadInputs: async (id) => {
    try {
      const res = await studio().comfy.uploadInputs(id)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  pullWorkflow: async (id) => {
    try {
      await studio().comfy.pullWorkflow(id)
    } catch {
      // best-effort sync — a transient failure shouldn't surface as an error
    }
  },

  saveLiveWorkflow: async (id, workflow) => {
    try {
      const res = await studio().comfy.saveLiveWorkflow(id, workflow)
      // Merge the updated frame so the inspector's "ready" state reflects the capture.
      if (res.ok && res.value) {
        const frame = res.value
        set((s) => ({ frames: s.frames.map((sh) => (sh.id === frame.id ? frame : sh)) }))
      }
    } catch {
      // best-effort autosave — a transient failure shouldn't surface as an error
    }
  },

  pullResult: async (id) => {
    set({ busyId: id, error: null })
    try {
      const res = await studio().comfy.pullLatest(id)
      if (!res.ok) return set({ error: res.error, busyId: null })
      const take = res.value
      set((s) => ({
        takesByFrame: { ...s.takesByFrame, [id]: [take, ...(s.takesByFrame[id] ?? [])] },
        frames: s.frames.map((sh) => (sh.id === id ? { ...sh, heroTakeId: take.id } : sh)),
        busyId: null,
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e), busyId: null })
    }
  },

  captureOutput: async (frameId, output) => {
    try {
      const res = await studio().comfy.captureOutput(frameId, output)
      if (!res.ok) return set({ error: res.error })
      const take = res.value
      set((s) => ({
        takesByFrame: { ...s.takesByFrame, [frameId]: [take, ...(s.takesByFrame[frameId] ?? [])] },
        frames: s.frames.map((sh) => (sh.id === frameId ? { ...sh, heroTakeId: take.id } : sh)),
      }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  rename: async (id, name) => {
    set((s) => ({ frames: s.frames.map((sh) => (sh.id === id ? { ...sh, name } : sh)) }))
    try {
      const res = await studio().frames.rename(id, name)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  reorder: async (orderedIds) => {
    set((s) => {
      const byId = new Map(s.frames.map((sh) => [sh.id, sh]))
      const next = orderedIds
        .map((id, i) => {
          const sh = byId.get(id)
          return sh ? { ...sh, position: i } : null
        })
        .filter((x): x is Frame => x !== null)
      return { frames: next }
    })
    try {
      const res = await studio().frames.reorder(orderedIds)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  remove: async (id) => {
    try {
      const res = await studio().frames.delete(id)
      if (!res.ok) return set({ error: res.error })
      set((s) => {
        const inputsByFrame = { ...s.inputsByFrame }
        const takesByFrame = { ...s.takesByFrame }
        delete inputsByFrame[id]
        delete takesByFrame[id]
        return {
          frames: s.frames.filter((sh) => sh.id !== id),
          inputsByFrame,
          takesByFrame,
          selectedId: s.selectedId === id ? null : s.selectedId,
        }
      })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  clone: async (id) => {
    try {
      const res = await studio().frames.clone(id)
      if (!res.ok) {
        set({ error: res.error })
        return null
      }
      await get().load() // bring in the new frame + its copied inputs
      return res.value
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
      return null
    }
  },

  unlink: async (id) => {
    try {
      const res = await studio().frames.unlink(id)
      if (!res.ok) return set({ error: res.error })
      const frame = res.value
      set((s) => ({ frames: s.frames.map((sh) => (sh.id === id ? frame : sh)) }))
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  exportFrames: async () => {
    set({ error: null, notice: null })
    try {
      const res = await studio().export.exportFrames()
      if (!res.ok) return set({ error: res.error })
      if (res.value === null) return // cancelled
      const { exported, skipped, dir } = res.value
      const skip = skipped.length > 0 ? `, ${skipped.length} skipped (no output)` : ''
      set({ notice: `Exported ${exported} frame${exported === 1 ? '' : 's'}${skip} → ${dir}` })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  select: (id) => set({ selectedId: id }),
  reset: () =>
    set({
      frames: [],
      inputsByFrame: {},
      takesByFrame: {},
      selectedId: null,
      busyId: null,
      error: null,
      notice: null,
    }),
}))
