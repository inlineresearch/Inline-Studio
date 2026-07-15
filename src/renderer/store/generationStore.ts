/**
 * State for the fal DAG generation engine: which frames are mid-run and their progress.
 * `run` kicks off a fire-and-forget run in main; progress/completion arrive via the
 * generation events (subscribed in MoodboardPanel), which call the setters here.
 */
import { create } from 'zustand'
import { getNodeDef } from '@shared/nodes/registry'
import { emptyResolvedInputs } from '@shared/nodes/types'
import type { FalRunRequest } from '@shared/ipc'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'
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
  /** The Core node (moodboard item) whose settings sidebar is open, or null. */
  settingsCoreItemId: string | null
  /** The model id whose info sidebar (inputs/outputs/cost) is open, or null. */
  infoModelId: string | null
  error: string | null

  /** Run just this fal node (its inputs use whatever upstream nodes already produced). */
  run: (frameId: string) => Promise<void>
  /** Run a Core workflow up to this canvas node (serializes its upstream closure). */
  runWorkflow: (itemId: string) => Promise<void>
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
  /** Open the Core-node settings sidebar for this item, or close it if already open for it. */
  toggleCoreSettings: (itemId: string) => void
  closeCoreSettings: () => void
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
  settingsCoreItemId: null,
  infoModelId: null,
  error: null,

  run: async (frameId) => {
    set((s) => ({
      error: null,
      busyByFrame: { ...s.busyByFrame, [frameId]: true },
      progressByFrame: { ...s.progressByFrame, [frameId]: 0 },
    }))
    const fail = (error: string): void =>
      set((s) => ({
        error,
        busyByFrame: { ...s.busyByFrame, [frameId]: false },
        progressByFrame: { ...s.progressByFrame, [frameId]: null },
      }))
    try {
      // Build the fal request client-side — fal node defs live in the browser. The web backend runs
      // exactly this request; the Electron backend ignores it and builds the request server-side.
      let request: FalRunRequest | undefined
      const frame = useFrameStore.getState().frames.find((f) => f.id === frameId)
      const def = frame?.modelId ? getNodeDef(frame.modelId) : undefined
      if (frame?.provider === 'fal' && def) {
        const resolved = await studio().frames.resolveFalInputs(frameId)
        if (!resolved.ok) return fail(resolved.error)
        if (!resolved.value.prompt) {
          return fail('Connect a Prompt node with some text to generate.')
        }
        const inputs = {
          ...emptyResolvedInputs(),
          images: resolved.value.images,
          videos: resolved.value.videos,
          audios: resolved.value.audios,
        }
        const runParams = { ...frame.params, prompt: resolved.value.prompt }
        request = {
          endpoint: def.resolveEndpoint(inputs),
          body: def.buildRequest(runParams, inputs),
          outputKind: def.outputKind,
        }
      }
      const res = await studio().generation.run(frameId, request)
      if (!res.ok) fail(res.error)
    } catch (e) {
      fail(ipcErrorMessage(e))
    }
  },

  runWorkflow: async (itemId) => {
    set((s) => ({
      error: null,
      busyByFrame: { ...s.busyByFrame, [itemId]: true },
      progressByFrame: { ...s.progressByFrame, [itemId]: 0 },
    }))
    try {
      const res = await studio().generation.runWorkflow(itemId)
      if (!res.ok) {
        set((s) => ({
          error: res.error,
          busyByFrame: { ...s.busyByFrame, [itemId]: false },
          progressByFrame: { ...s.progressByFrame, [itemId]: null },
        }))
      }
    } catch (e) {
      set((s) => ({
        error: ipcErrorMessage(e),
        busyByFrame: { ...s.busyByFrame, [itemId]: false },
        progressByFrame: { ...s.progressByFrame, [itemId]: null },
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
      await studio().generation.cancel(frameId)
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  resumePending: async () => {
    try {
      await studio().generation.resumePending()
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
      const res = await studio().frames.setFalParams(frameId, params)
      if (!res.ok) set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  setModel: async (frameId, modelId) => {
    try {
      const res = await studio().frames.setModel(frameId, modelId)
      if (!res.ok) return set({ error: res.error })
      // Output kind + params changed — refresh the frame store so the node re-resolves.
      await useFrameStore.getState().load()
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    }
  },

  // The fal-settings, Core-settings, and model-info sidebars share the right gutter, so opening any
  // one closes the others.
  toggleSettings: (frameId) =>
    set((s) => ({
      settingsFrameId: s.settingsFrameId === frameId ? null : frameId,
      settingsCoreItemId: null,
      infoModelId: null,
    })),
  closeSettings: () => set({ settingsFrameId: null }),
  toggleCoreSettings: (itemId) =>
    set((s) => ({
      settingsCoreItemId: s.settingsCoreItemId === itemId ? null : itemId,
      settingsFrameId: null,
      infoModelId: null,
    })),
  closeCoreSettings: () => set({ settingsCoreItemId: null }),
  openModelInfo: (modelId) =>
    set({ infoModelId: modelId, settingsFrameId: null, settingsCoreItemId: null }),
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
