/**
 * The run target(s) of the currently-selected graph - the output node(s) that show the Run control.
 * MoodboardPanel computes this from the selection + connectors; each generation node reads whether
 * it is a target (`useIsRunTarget`) to decide whether to float its Run/Stop pill.
 */
import { create } from 'zustand'

interface GraphSelectionState {
  /** Node/item ids that are a selected graph's output node. */
  runTargets: string[]
  setRunTargets: (ids: string[]) => void
}

export const useGraphSelectionStore = create<GraphSelectionState>((set) => ({
  runTargets: [],
  setRunTargets: (ids) =>
    set((s) => {
      // Only write on a real change, so unrelated selection churn doesn't re-render every node.
      if (s.runTargets.length === ids.length && s.runTargets.every((id, i) => id === ids[i])) {
        return s
      }
      return { runTargets: ids }
    }),
}))

/** Whether `id` is a run target of the selected graph (a boolean selector, so a node only re-renders
 * when its own target status flips). */
export function useIsRunTarget(id: string): boolean {
  return useGraphSelectionStore((s) => s.runTargets.includes(id))
}
