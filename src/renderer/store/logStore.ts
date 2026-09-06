/** Lines streamed by a run, keyed by run id, for whatever Logger node is wired to that run. */
import { create } from 'zustand'
import { studio } from '../lib/studio'

/** Deep enough to still hold a run's setup lines once a few hundred have streamed in, bounded so a
 *  long run cannot grow it without limit. Same depth the Trainer's own buffer uses. */
const MAX_LINES = 3000

interface LogState {
  linesByRun: Record<string, string[]>
  append: (runId: string, line: string) => void
  clear: (runId: string) => void
}

export const useLogStore = create<LogState>((set) => ({
  linesByRun: {},
  append: (runId, line) =>
    set((s) => ({
      linesByRun: {
        ...s.linesByRun,
        // One line per entry: the sender owns its own formatting, so a multi-line write splits here
        // rather than rendering as one unscrollable block.
        [runId]: [...(s.linesByRun[runId] ?? []), ...line.split('\n')].slice(-MAX_LINES),
      },
    })),
  clear: (runId) =>
    set((s) => {
      const next = { ...s.linesByRun }
      delete next[runId]
      return { linesByRun: next }
    }),
}))

/** Every stream a Logger can show. Training is here so the node is useful against a real run today. */
export function subscribeLogEvents(): () => void {
  const append = useLogStore.getState().append
  const unsubs = [studio().events.onTrainingLog((e) => append(e.runId, e.line))]
  return () => unsubs.forEach((u) => u())
}
