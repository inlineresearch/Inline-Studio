/**
 * State for the LoRA Trainer tab: datasets, their items, training runs, and the live run telemetry.
 * Mutations go through `studio().training.*`; live progress/samples/host-stats arrive via the
 * training events (subscribed by `subscribeTrainingEvents`, wired in TrainerPanel) which call the
 * setters here. Mirrors `generationStore`'s shape.
 */
import { create } from 'zustand'
import type {
  SystemStatsEvent,
  TrainingDataset,
  TrainingDatasetItem,
  TrainingDoneEvent,
  TrainingErrorEvent,
  TrainingHyperparams,
  TrainingProgressEvent,
  TrainingRun,
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
  error: string | null

  progressByRun: Record<string, RunProgress>
  lossByRun: Record<string, number[]>
  samplesByRun: Record<string, string[]>
  systemStats: SystemStatsEvent | null

  loadDatasets: () => Promise<void>
  createDataset: (name: string, triggerWord: string) => Promise<TrainingDataset | null>
  selectDataset: (datasetId: string | null) => void
  loadItems: (datasetId: string) => Promise<void>
  addItems: (datasetId: string, assetIds: string[]) => Promise<void>
  removeItem: (datasetId: string, itemId: string) => Promise<void>
  setCaption: (datasetId: string, itemId: string, caption: string) => Promise<void>
  autoCaption: (datasetId: string, overwrite: boolean) => Promise<void>
  loadRuns: () => Promise<void>
  start: (datasetId: string, hyperparams: TrainingHyperparams) => Promise<void>
  resume: (runId: string) => Promise<void>
  cancel: (runId: string) => Promise<void>

  applyProgress: (e: TrainingProgressEvent) => void
  applySample: (e: TrainingSampleEvent) => void
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
  error: null,
  progressByRun: {},
  lossByRun: {},
  samplesByRun: {},
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
    if (!res.ok) return set({ error: res.error })
    await get().loadItems(datasetId)
  },

  removeItem: async (datasetId, itemId) => {
    const res = await studio().training.removeItem(itemId)
    if (!res.ok) return set({ error: res.error })
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

  autoCaption: async (datasetId, overwrite) => {
    set({ captioning: true, error: null })
    try {
      const res = await studio().training.autoCaption(datasetId, overwrite)
      if (res.ok) set((s) => ({ itemsByDataset: { ...s.itemsByDataset, [datasetId]: res.value } }))
      else set({ error: res.error })
    } catch (e) {
      set({ error: ipcErrorMessage(e) })
    } finally {
      set({ captioning: false })
    }
  },

  loadRuns: async () => {
    const res = await studio().training.listRuns()
    if (res.ok) set({ runs: res.value })
    else set({ error: res.error })
  },

  start: async (datasetId, hyperparams) => {
    const res = await studio().training.start(datasetId, hyperparams)
    if (!res.ok) return set({ error: res.error })
    set((s) => ({ runs: [res.value, ...s.runs.filter((r) => r.id !== res.value.id)] }))
  },

  resume: async (runId) => {
    const res = await studio().training.resume(runId)
    if (!res.ok) return set({ error: res.error })
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

  applyError: (e) =>
    set((s) => ({
      runs: s.runs.map((r) => (r.id === e.runId ? { ...r, error: e.error } : r)),
    })),

  applyStats: (systemStats) => set({ systemStats }),
  setError: (error) => set({ error }),
}))

/** Wire the training + telemetry events to the store. Returns an unsubscribe fn. */
export function subscribeTrainingEvents(): () => void {
  const s = useTrainingStore.getState()
  const unsubs = [
    studio().events.onTrainingProgress((e) => s.applyProgress(e)),
    studio().events.onTrainingSample((e) => s.applySample(e)),
    studio().events.onTrainingDone((e) => s.applyDone(e)),
    studio().events.onTrainingError((e) => s.applyError(e)),
    studio().events.onSystemStats((e) => s.applyStats(e)),
  ]
  return () => unsubs.forEach((u) => u())
}
