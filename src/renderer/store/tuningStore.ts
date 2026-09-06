/** Reference sweeps in flight, and what the finished ones found. */
import { create } from 'zustand'
import type { SweepResult, TuneProgressEvent } from '@shared/types'
import { studio } from '../lib/studio'
import { useLogStore } from './logStore'
import { useMoodboardStore } from './moodboardStore'

interface TuningState {
  progressByRun: Record<string, TuneProgressEvent>
  resultByRun: Record<string, SweepResult>
  errorByRun: Record<string, string>
  /** Fetch a run the live events already missed, which is every run after a reload. */
  load: (runId: string) => Promise<void>
}

const fetching = new Set<string>()

export const useTuningStore = create<TuningState>((set, get) => ({
  progressByRun: {},
  resultByRun: {},
  errorByRun: {},
  load: async (runId) => {
    if (get().resultByRun[runId] || fetching.has(runId)) return
    fetching.add(runId)
    const res = await studio().characters.sweepResult(runId)
    fetching.delete(runId)
    if (res.ok) set((s) => ({ resultByRun: { ...s.resultByRun, [runId]: res.value } }))
  },
}))

export function subscribeTuningEvents(): () => void {
  const unsubs = [
    studio().events.onTuneProgress((e) =>
      useTuningStore.setState((s) => ({ progressByRun: { ...s.progressByRun, [e.runId]: e } })),
    ),
    // The sweep's own lines go to the shared log buffer, so a wired Logger shows them with no
    // knowledge of what a sweep is.
    studio().events.onTuneLog((e) => useLogStore.getState().append(e.runId, e.line)),
    studio().events.onTuneDone((e) =>
      useTuningStore.setState((s) => ({ resultByRun: { ...s.resultByRun, [e.runId]: e.result } })),
    ),
    studio().events.onTuneError((e) =>
      useTuningStore.setState((s) => ({ errorByRun: { ...s.errorByRun, [e.runId]: e.error } })),
    ),
    // A sweep node learns its run id here; the board holds it so a reload still finds the stream.
    studio().events.onTuneNodeBound((e) => {
      void useMoodboardStore.getState().patchItemData(e.itemId, { runId: e.runId })
    }),
  ]
  return () => unsubs.forEach((u) => u())
}
