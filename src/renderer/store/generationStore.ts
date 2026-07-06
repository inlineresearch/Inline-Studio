/**
 * State for the fal DAG generation engine: which frames are mid-run and their progress.
 * `run` kicks off a fire-and-forget run in main; progress/completion arrive via the
 * generation events (subscribed in MoodboardPanel), which call the setters here.
 */
import { create } from 'zustand'
import { ipcErrorMessage } from '../lib/ipcError'
import { useFrameStore } from './frameStore'

interface GenerationState {
  /** Per-frame "currently generating" flag. */
  busyByFrame: Record<string, boolean>
  /** Per-frame progress 0..1, or null when idle. */
  progressByFrame: Record<string, number | null>
  /** Per-frame short status label from the fal queue. */
  statusByFrame: Record<string, string | undefined>
  /** The frame whose settings sidebar is open, or null. */
  settingsFrameId: string | null
  /** The model id whose info sidebar (inputs/outputs/cost) is open, or null. */
  infoModelId: string | null
  error: string | null

  /** Run just this fal node (its inputs use whatever upstream nodes already produced). */
  run: (frameId: string) => Promise<void>
  /** Abort a frame's run (or all when no id) — resets its node immediately. */
  cancel: (frameId?: string) => Promise<void>
  /** Ask main to re-poll + finish any generations left in flight from a previous session. */
  resumePending: () => Promise<void>
  /** Persist a fal frame's param values (optimistically updates the frame store). */
  setParams: (frameId: string, params: Record<string, unknown>) => Promise<void>
  /** Switch a fal frame to a different model (resets params + output kind). */
  setModel: (frameId: string, modelId: string) => Promise<void>
  /** Open the settings sidebar for this frame, or close it if it's already open for this frame. */
  toggleSettings: (frameId: string) => void
  closeSettings: () => void
  /** Open the model-info sidebar (inputs/outputs/cost) for a model id. */
  openModelInfo: (modelId: string) => void
  closeModelInfo: () => void
  setProgress: (frameId: string, fraction: number | null, status?: string) => void
  setBusy: (frameId: string, busy: boolean) => void
  /** Clear all busy/progress (call when a run finishes or fails). */
  finishAll: () => void
  setError: (error: string | null) => void
}

export const useGenerationStore = create<GenerationState>((set) => ({
  busyByFrame: {},
  progressByFrame: {},
  statusByFrame: {},
  settingsFrameId: null,
  infoModelId: null,
  error: null,

  run: async (frameId) => {
    set((s) => ({
      error: null,
      busyByFrame: { ...s.busyByFrame, [frameId]: true },
      progressByFrame: { ...s.progressByFrame, [frameId]: 0 },
    }))
    try {
      const res = await window.inlineStudio.generation.run(frameId)
      if (!res.ok) {
        set((s) => ({
          error: res.error,
          busyByFrame: { ...s.busyByFrame, [frameId]: false },
          progressByFrame: { ...s.progressByFrame, [frameId]: null },
        }))
      }
    } catch (e) {
      set((s) => ({
        error: ipcErrorMessage(e),
        busyByFrame: { ...s.busyByFrame, [frameId]: false },
        progressByFrame: { ...s.progressByFrame, [frameId]: null },
      }))
    }
  },

  cancel: async (frameId) => {
    // Reset the node right away — the run is also cancelled server-side, but the UI shouldn't
    // wait on the round-trip. Without a frame id, clear every in-flight node.
    set((s) =>
      frameId
        ? {
            busyByFrame: { ...s.busyByFrame, [frameId]: false },
            progressByFrame: { ...s.progressByFrame, [frameId]: null },
            statusByFrame: { ...s.statusByFrame, [frameId]: undefined },
          }
        : { busyByFrame: {}, progressByFrame: {}, statusByFrame: {} },
    )
    try {
      await window.inlineStudio.generation.cancel(frameId)
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  resumePending: async () => {
    try {
      await window.inlineStudio.generation.resumePending()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setParams: async (frameId, params) => {
    // Optimistic: reflect the edit on the frame immediately so the node re-renders.
    useFrameStore.setState((s) => ({
      frames: s.frames.map((f) => (f.id === frameId ? { ...f, params } : f)),
    }))
    try {
      const res = await window.inlineStudio.frames.setFalParams(frameId, params)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setModel: async (frameId, modelId) => {
    try {
      const res = await window.inlineStudio.frames.setModel(frameId, modelId)
      if (!res.ok) return set({ error: res.error })
      // Output kind + params changed — refresh the frame store so the node re-resolves.
      await useFrameStore.getState().load()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  // The settings and model-info sidebars share the right gutter, so opening one closes the other.
  toggleSettings: (frameId) =>
    set((s) => ({
      settingsFrameId: s.settingsFrameId === frameId ? null : frameId,
      infoModelId: null,
    })),
  closeSettings: () => set({ settingsFrameId: null }),
  openModelInfo: (modelId) => set({ infoModelId: modelId, settingsFrameId: null }),
  closeModelInfo: () => set({ infoModelId: null }),

  setProgress: (frameId, fraction, status) =>
    set((s) => ({
      progressByFrame: { ...s.progressByFrame, [frameId]: fraction },
      statusByFrame: { ...s.statusByFrame, [frameId]: status },
    })),

  setBusy: (frameId, busy) => set((s) => ({ busyByFrame: { ...s.busyByFrame, [frameId]: busy } })),

  finishAll: () => set({ busyByFrame: {}, progressByFrame: {}, statusByFrame: {} }),

  setError: (error) => set({ error }),
}))
