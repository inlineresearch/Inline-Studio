/**
 * State for the LoRA Trainer tab: datasets, their items, training runs, and the live run telemetry.
 * Mutations go through `studio().training.*`; live progress/samples/host-stats arrive via the
 * training events (subscribed by `subscribeTrainingEvents`, wired in TrainerPanel) which call the
 * setters here. Mirrors `generationStore`'s shape.
 */
import { create } from 'zustand'
import type {
  SystemStatsEvent,
  CaptionerModel,
  TrainingDataset,
  TrainingDatasetItem,
  TrainingDoneEvent,
  TrainingErrorEvent,
  TrainingHyperparams,
  TrainingProgressEvent,
  TrainingRun,
  CaptionProgressEvent,
  TrainingLogEvent,
  TrainingSampleEvent,
} from '@shared/types'
import { studio } from '@/lib/studio'
import { ipcErrorMessage } from '../lib/ipcError'

/** Live progress for one run, updated from `onTrainingProgress`. */
export interface RunProgress {
  fraction: number
  step: number
  totalSteps: number
  status?: string
}

interface TrainingState {
  datasets: TrainingDataset[]
  itemsByDataset: Record<string, TrainingDatasetItem[]>
  runs: TrainingRun[]
  activeDatasetId: string | null
  captioning: boolean
  captioners: CaptionerModel[]
  error: string | null

  progressByRun: Record<string, RunProgress>
  lossByRun: Record<string, number[]>
  /** The trainer subprocess's streamed stdout, per run - shown in the Trainer node. */
  logsByRun: Record<string, string[]>
  /** Live auto-caption progress per dataset, cleared when the captioner exits. */
  captionProgress: Record<string, { done: number; total: number }>
  samplesByRun: Record<string, string[]>
  systemStats: SystemStatsEvent | null

  loadDatasets: () => Promise<void>
  createDataset: (name: string, triggerWord: string) => Promise<TrainingDataset | null>
  selectDataset: (datasetId: string | null) => void
  loadItems: (datasetId: string) => Promise<void>
  /** Returns the created items, so a caller can pair captions to the assets it just added. */
  addItems: (datasetId: string, assetIds: string[]) => Promise<TrainingDatasetItem[]>
  /** Import a folder on the Core machine, sidecar captions included. Returns an error string. */
  addFromPath: (datasetId: string, path: string) => Promise<string | null>
  removeItem: (datasetId: string, itemId: string) => Promise<void>
  /** Remove every image from a dataset in one go, refetching once when done. */
  removeAll: (datasetId: string) => Promise<void>
  setCaption: (datasetId: string, itemId: string, caption: string) => Promise<void>
  autoCaption: (datasetId: string, overwrite: boolean, model?: string) => Promise<void>
  /** The caption models Core offers; loaded once, lazily. */
  loadCaptioners: () => Promise<void>
  loadRuns: () => Promise<void>
  /** Returns the created run so a canvas node can persist its `runId` and rebind after a reload. */
  start: (datasetId: string, hyperparams: TrainingHyperparams) => Promise<TrainingRun | null>
  resume: (runId: string) => Promise<void>
  cancel: (runId: string) => Promise<void>
  /** Drop a run's checkpoints so a changed configuration starts clean. */
  discard: (runId: string) => Promise<void>

  applyProgress: (e: TrainingProgressEvent) => void
  applySample: (e: TrainingSampleEvent) => void
  applyLog: (e: TrainingLogEvent) => void
  applyCaptionProgress: (e: CaptionProgressEvent) => void
  applyDone: (e: TrainingDoneEvent) => void
  applyError: (e: TrainingErrorEvent) => void
  applyStats: (e: SystemStatsEvent) => void
  setError: (error: string | null) => void
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  datasets: [],
  itemsByDataset: {},
  runs: [],
  activeDatasetId: null,
  captioning: false,
  captioners: [],
  error: null,
  progressByRun: {},
  lossByRun: {},
  samplesByRun: {},
  logsByRun: {},
  captionProgress: {},
  systemStats: null,

  loadDatasets: async () => {
    const res = await studio().training.listDatasets()
    if (res.ok) set({ datasets: res.value })
    else set({ error: res.error })
  },

  createDataset: async (name, triggerWord) => {
    const res = await studio().training.createDataset({ name, triggerWord })
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    set((s) => ({ datasets: [res.value, ...s.datasets], activeDatasetId: res.value.id }))
    return res.value
  },

  selectDataset: (activeDatasetId) => {
    set({ activeDatasetId })
    if (activeDatasetId) void get().loadItems(activeDatasetId)
  },

  loadItems: async (datasetId) => {
    const res = await studio().training.listItems(datasetId)
    if (res.ok) set((s) => ({ itemsByDataset: { ...s.itemsByDataset, [datasetId]: res.value } }))
    else set({ error: res.error })
  },

  addItems: async (datasetId, assetIds) => {
    const res = await studio().training.addItems(datasetId, assetIds)
    if (!res.ok) {
      set({ error: res.error })
      return []
    }
    await get().loadItems(datasetId)
    return res.value
  },

  addFromPath: async (datasetId, path) => {
    const res = await studio().training.addFromPath(datasetId, path)
    if (!res.ok) {
      // Returned rather than only stored: a bad path is the common case here and the field wants
      // to show it inline, next to what the reader typed.
      set({ error: res.error })
      return ipcErrorMessage(res.error)
    }
    await get().loadItems(datasetId)
    return null
  },

  removeItem: async (datasetId, itemId) => {
    const res = await studio().training.removeItem(itemId)
    if (!res.ok) return set({ error: res.error })
    await get().loadItems(datasetId)
  },

  removeAll: async (datasetId) => {
    const ids = (get().itemsByDataset[datasetId] ?? []).map((it) => it.id)
    const results = await Promise.all(ids.map((id) => studio().training.removeItem(id)))
    const failed = results.find((r) => !r.ok)
    if (failed && !failed.ok) set({ error: failed.error })
    await get().loadItems(datasetId)
  },

  setCaption: async (datasetId, itemId, caption) => {
    const res = await studio().training.setCaption(itemId, caption)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({
      itemsByDataset: {
        ...s.itemsByDataset,
        [datasetId]: (s.itemsByDataset[datasetId] ?? []).map((it) =>
          it.id === itemId ? res.value : it,
        ),
      },
    }))
  },

  autoCaption: async (datasetId, overwrite, model) => {
    set({ captioning: true, error: null })
    try {
      const res = await studio().training.autoCaption(datasetId, overwrite, model)
      if (res.ok) set((s) => ({ itemsByDataset: { ...s.itemsByDataset, [datasetId]: res.value } }))
      else set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    } finally {
      set({ captioning: false })
    }
  },

  loadCaptioners: async () => {
    if (get().captioners.length) return
    const res = await studio().training.captioners()
    if (res.ok) set({ captioners: res.value })
  },

  loadRuns: async () => {
    const res = await studio().training.listRuns()
    if (res.ok) set({ runs: res.value })
    else set({ error: res.error })
  },

  start: async (datasetId, hyperparams) => {
    const res = await studio().training.start(datasetId, hyperparams)
    if (!res.ok) {
      set({ error: res.error })
      return null
    }
    set((s) => ({ runs: [res.value, ...s.runs.filter((r) => r.id !== res.value.id)] }))
    return res.value
  },

  resume: async (runId) => {
    const res = await studio().training.resume(runId)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({ runs: s.runs.map((r) => (r.id === runId ? res.value : r)) }))
  },

  discard: async (runId) => {
    const res = await studio().training.discard(runId)
    if (!res.ok) return
    set((s) => ({ runs: s.runs.map((r) => (r.id === runId ? res.value : r)) }))
  },
  cancel: async (runId) => {
    const res = await studio().training.cancel(runId)
    if (!res.ok) set({ error: res.error })
  },

  applyProgress: (e) =>
    set((s) => ({
      progressByRun: {
        ...s.progressByRun,
        [e.runId]: {
          fraction: e.fraction,
          step: e.step,
          totalSteps: e.totalSteps,
          status: e.status,
        },
      },
      lossByRun:
        typeof e.loss === 'number'
          ? { ...s.lossByRun, [e.runId]: [...(s.lossByRun[e.runId] ?? []), e.loss] }
          : s.lossByRun,
    })),

  applySample: (e) =>
    set((s) => ({
      samplesByRun: { ...s.samplesByRun, [e.runId]: [...(s.samplesByRun[e.runId] ?? []), e.path] },
    })),

  applyLog: (e) =>
    set((s) => ({
      // Capped so a long run can't grow the buffer without bound; the node shows the tail. Deep
      // enough to still hold the setup and precache lines once a few hundred steps have streamed
      // in, since those are what a "Copy logs" for a failed run needs to include.
      logsByRun: {
        ...s.logsByRun,
        [e.runId]: [...(s.logsByRun[e.runId] ?? []), e.line].slice(-3000),
      },
    })),

  applyCaptionProgress: (e) =>
    set((s) => {
      const next = { ...s.captionProgress }
      if (e.finished) delete next[e.datasetId]
      else next[e.datasetId] = { done: e.done, total: e.total }
      return { captionProgress: next }
    }),

  applyDone: (e) => {
    set((s) => ({
      runs: s.runs.map((r) =>
        r.id === e.runId
          ? { ...r, status: 'done', outputLoraPath: e.outputLoraPath, progressFraction: 1 }
          : r,
      ),
    }))
    void get().loadRuns()
  },

  applyError: (e) => {
    set((s) => ({
      runs: s.runs.map((r) => (r.id === e.runId ? { ...r, error: e.error } : r)),
    }))
    // A run ending (cancel/crash) only broadcasts the error - reload so its new status lands,
    // otherwise a stopped run still reads as "training" and its Stop control never flips back.
    void get().loadRuns()
  },

  applyStats: (systemStats) => set({ systemStats }),
  setError: (error) => set({ error }),
}))

/** Wire the training + telemetry events to the store. Returns an unsubscribe fn. */
export function subscribeTrainingEvents(): () => void {
  const s = useTrainingStore.getState()
  const unsubs = [
    studio().events.onTrainingProgress((e) => s.applyProgress(e)),
    studio().events.onTrainingSample((e) => s.applySample(e)),
    studio().events.onTrainingLog((e) => s.applyLog(e)),
    studio().events.onCaptionProgress((e) => s.applyCaptionProgress(e)),
    studio().events.onTrainingDone((e) => s.applyDone(e)),
    studio().events.onTrainingError((e) => s.applyError(e)),
    studio().events.onSystemStats((e) => s.applyStats(e)),
  ]
  return () => unsubs.forEach((u) => u())
}
