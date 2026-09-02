/**
 * Open the Workflows popup once, on the first empty project the user lands in.
 *
 * Gated on the catalogue actually resolving with entries: a fresh offline install would otherwise
 * be greeted by an empty dialog it never asked for.
 */
import { useEffect, useRef } from 'react'
import { studio } from '@/lib/studio'
import { useProjectStore } from '../../store/projectStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useWorkflowsStore } from '../../store/workflowsStore'

export function useWorkflowsAutoOpen(): void {
  const project = useProjectStore((s) => s.current)
  const items = useMoodboardStore((s) => s.items)
  const loading = useMoodboardStore((s) => s.loading)
  const considered = useRef<string | null>(null)

  useEffect(() => {
    const projectId = project?.id
    if (!projectId || loading || project.workflowsPrompted) return
    // `items` is briefly empty while the board loads, so a non-empty board must settle first.
    if (items.length > 0) {
      considered.current = projectId
      return
    }
    if (considered.current === projectId) return
    considered.current = projectId

    void useWorkflowsStore
      .getState()
      .load()
      .then(() => {
        if (!useWorkflowsStore.getState().catalogue?.entries.length) return
        useWorkflowsStore.getState().setOpen(true)
        // Marked only once it was actually shown: burning the flag on an offline first launch
        // would mean this project never gets the popup at all.
        void studio().workflows.markPrompted()
      })
  }, [project, items.length, loading])
}
