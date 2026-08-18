/**
 * State for the fal DAG generation engine: which frames are mid-run and their progress.
 * `run` kicks off a fire-and-forget run in main; progress/completion arrive via the
 * generation events (subscribed in MoodboardPanel), which call the setters here.
 */
import { create } from 'zustand'
import { getNodeDef } from '@shared/nodes/registry'
import { emptyResolvedInputs, mediaFamily, portMedia } from '@shared/nodes/types'
import type { FalRunRequest } from '@shared/ipc'
import { ipcErrorMessage } from '../lib/ipcError'
import { studio } from '@/lib/studio'
import { useFrameStore } from './frameStore'
import { useMoodboardStore } from './moodboardStore'
import { useModelRequirementsStore } from './modelRequirementsStore'

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
  /** Abort a frame's run (or all when no id) - resets its node immediately. */
  cancel: (frameId?: string) => Promise<void>
  /** Ask main to re-poll + finish any generations left in flight from a previous session. */
  resumePending: () => Promise<void>
  /** Rebuild the queue from what Core still has running, after a page refresh threw ours away. */
  hydrateActive: () => Promise<void>
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
  /**
   * Clear one node's busy/progress. Keyed by id rather than clearing everything, because with a
   * visible queue more than one node is legitimately in flight at a time.
   */
  finishRun: (frameId: string) => void
  /** Drop all local node state (project close). The runs themselves keep going server-side. */
  reset: () => void
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
      // Build the fal request client-side - fal node defs live in the browser. The web backend runs
      // exactly this request; the Electron backend ignores it and builds the request server-side.
      let request: FalRunRequest | undefined
      const frame = useFrameStore.getState().frames.find((f) => f.id === frameId)
      const def = frame?.modelId ? getNodeDef(frame.modelId) : undefined
      if (frame?.provider === 'fal' && def) {
        const resolved = await studio().frames.resolveFalInputs(frameId)
        if (!resolved.ok) return fail(resolved.error)
        // Most models need a Prompt node; a few (e.g. Sonilo video→music) mark it optional.
        if (!def.promptOptional && !resolved.value.prompt) {
          return fail('Connect a Prompt node with some text to generate.')
        }
        const inputs = {
          ...emptyResolvedInputs(),
          images: resolved.value.images,
          videos: resolved.value.videos,
          audios: resolved.value.audios,
          byHandle: resolved.value.byHandle ?? {},
        }
        // A required media port with nothing behind it would go out as an empty URL, and fal
        // answers that with a 422 "failed to download the file" that says nothing about the cause.
        const starved = def.inputs.filter(
          (p) => p.required && mediaFamily(p.kind) && portMedia(def, inputs, p.id).length === 0,
        )
        if (starved.length > 0) {
          return fail(
            `${def.title} needs ${starved.map((p) => p.label.toLowerCase()).join(' and ')}. ` +
              'Wire it into the node, or drop the media onto the node itself.',
          )
        }
        const runParams = { ...frame.params, prompt: resolved.value.prompt ?? '' }
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
    // The node stays busy and says "Stopping…" until Core confirms. Cancellation is cooperative:
    // the run stops at its next checkpoint, and inside a model load that is seconds away. Clearing
    // the node here used to claim it had stopped while the GPU was still working.
    const stopping = 'Stopping…'
    set((s) =>
      frameId
        ? { statusByFrame: { ...s.statusByFrame, [frameId]: stopping } }
        : {
            statusByFrame: Object.fromEntries(
              Object.keys(s.busyByFrame)
                .filter((id) => s.busyByFrame[id])
                .map((id) => [id, stopping]),
            ),
          },
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

  hydrateActive: async () => {
    try {
      const res = await studio().generation.active()
      if (!res.ok) return
      // Core's answer REPLACES local state rather than merging into it. Restarting Core kills every
      // run, and a merge left the nodes spinning forever because nothing ever said they stopped.
      // Anything absent here is not running, whatever this tab last saw.
      const busy: Record<string, boolean> = {}
      const progress: Record<string, number | null> = {}
      const status: Record<string, string | undefined> = {}
      for (const run of res.value) {
        busy[run.frameId] = true
        progress[run.frameId] = run.fraction
        status[run.frameId] = run.status
      }
      set({ busyByFrame: busy, progressByFrame: progress, statusByFrame: status })
    } catch {
      // A backend that predates the channel simply has no queue to restore.
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
      // Output kind + params changed - refresh the frame store so the node re-resolves.
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

  finishRun: (frameId) =>
    set((s) => ({
      busyByFrame: { ...s.busyByFrame, [frameId]: false },
      progressByFrame: { ...s.progressByFrame, [frameId]: null },
      statusByFrame: { ...s.statusByFrame, [frameId]: undefined },
    })),

  reset: () => set({ busyByFrame: {}, progressByFrame: {}, statusByFrame: {}, error: null }),

  setError: (error) => set({ error }),
}))

/**
 * Subscribe once for the whole app (see App.tsx). This used to live inside MoodboardPanel, which
 * unmounts on a tab switch - so switching to the Trainer mid-render dropped every progress event.
 */
/** True when the error was a missing-model one and the popup took over from it. */
function openMissingModels(itemId: string, error: string): boolean {
  if (!/models missing/i.test(error)) return false
  const item = useMoodboardStore.getState().items.find((i) => i.id === itemId)
  const nodeType = item?.type === 'core' ? item.data.core?.type : undefined
  if (!nodeType) return false
  void useModelRequirementsStore
    .getState()
    .checkOnUse(nodeType, 'This node needs models before it can generate.')
  return true
}

export function subscribeGenerationEvents(): () => void {
  const gen = useGenerationStore.getState()
  // Finish any generations still running when the app last closed; their events arrive below.
  void gen.resumePending()
  // A page refresh throws away this tab's copy of the queue while Core keeps working, and a run
  // inside a long model load emits nothing for minutes, so waiting for an event is not enough.
  void gen.hydrateActive()
  const unsubs = [
    studio().events.onGenerationProgress((e) => {
      gen.setBusy(e.frameId, true)
      gen.setProgress(e.frameId, e.fraction, e.status)
    }),
    studio().events.onGenerationNodeDone((e) => {
      gen.setBusy(e.frameId, false)
      gen.setProgress(e.frameId, null)
      void useFrameStore.getState().load()
    }),
    studio().events.onGenerationDone((e) => {
      gen.finishRun(e.targetFrameId)
      void useFrameStore.getState().load()
      void useMoodboardStore.getState().load()
    }),
    studio().events.onGenerationError((e) => {
      gen.finishRun(e.frameId ?? e.targetFrameId)
      // A missing weight file is a thing to fix, not a sentence to read: open the popup that can
      // fetch it rather than restating the filenames as an error.
      if (openMissingModels(e.frameId ?? e.targetFrameId, e.error)) return
      gen.setError(e.error)
    }),
    studio().events.onGenerationCancelled((e) => {
      gen.finishRun(e.targetFrameId)
    }),
  ]
  return () => unsubs.forEach((u) => u())
}
