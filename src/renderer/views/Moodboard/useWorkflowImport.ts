/**
 * Importing a published workflow onto the canvas: build, select, frame.
 *
 * Sample inputs are deliberately not fetched: media-bearing nodes land as empty Load Assets nodes
 * for the user to drop their own media onto.
 *
 * The selection step is not cosmetic. `toNodes` rebuilds every node without `selected` whenever
 * `items` changes, so selecting has to happen *after* the new items land - hence the pending ref
 * plus an effect keyed on items rather than a straight call.
 */
import { useCallback, useEffect, useState } from 'react'
import type { Node } from '@xyflow/react'
import type { WorkflowDetail } from '@shared/types'
import type { Recipe } from '../../lib/pngRecipe'
import { buildGraphFromRecipe } from '../../lib/recipeGraph'
import { checkGraphModels } from '../../lib/checkModels'
import { boundsOf, placeImport, recipeTarget } from '../../lib/graphPlacement'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useOnboardingStore } from '../../store/onboardingStore'
import { useGenerationStore } from '../../store/generationStore'
import { useWorkflowsStore } from '../../store/workflowsStore'
import { studio } from '@/lib/studio'

interface Options {
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>
  fitBounds: (
    bounds: { x: number; y: number; width: number; height: number },
    opts?: { padding?: number; duration?: number },
  ) => void
  centre: () => { x: number; y: number }
}

export function useWorkflowImport({ setNodes, fitBounds, centre }: Options): {
  importWorkflow: (detail: WorkflowDetail) => void
} {
  const items = useMoodboardStore((s) => s.items)
  const [importing, setImporting] = useState(false)
  // State, not a ref: the last node lands in `items` before the build resolves, so a ref set
  // afterwards would never re-run this effect and the view would never travel to the new graph.
  const [pending, setPending] = useState<string[] | null>(null)

  useEffect(() => {
    if (!pending) return
    const landed = items.filter((i) => pending.includes(i.id))
    if (landed.length !== pending.length) return

    const frame = boundsOf(landed)
    setPending(null)
    setNodes((nodes) => nodes.map((n) => ({ ...n, selected: pending.includes(n.id) })))
    // fitBounds over fitView: fitView reads React Flow's measured sizes, which are still zero on
    // the render the nodes first appear in, so an import could frame an empty box at the origin.
    if (frame) fitBounds(frame, { padding: 0.3, duration: 400 })
  }, [items, pending, setNodes, fitBounds])

  const importWorkflow = useCallback(
    (detail: WorkflowDetail): void => {
      if (importing) return
      const recipe = detail.graph as Recipe | null
      const graphItems = recipe?.graph?.items
      if (!graphItems?.length) {
        useGenerationStore.getState().setError(`${detail.title} has no graph to import.`)
        return
      }

      const target = recipeTarget(graphItems, recipe?.target)
      if (!target) return
      const drop = placeImport(items, graphItems, target, centre())

      setImporting(true)
      useWorkflowsStore.getState().setImporting(detail.slug)
      // Closed before the build, not after: a graph takes a couple of seconds of round trips to
      // rebuild, and watching the nodes appear reads as progress where a frozen dialog reads as a
      // hang.
      useWorkflowsStore.getState().setOpen(false)

      void buildGraphFromRecipe(recipe as Recipe, drop)
        .then(({ ids, targetId }) => {
          // A build that produced nothing must not move the public counter.
          if (!ids.length) {
            useGenerationStore.getState().setError(`${detail.title} could not be rebuilt.`)
            return
          }
          setPending(ids)
          void studio().workflows.event(detail.slug, 'import')
          // After placing, not before: the graph is worth having even when a weight file is missing.
          void checkGraphModels(recipe, `${detail.title} was imported.`).then((missing) => {
            // A Run hint is useless until Run works, so it waits on there being nothing to download.
            if (missing === 0 && targetId)
              useOnboardingStore.getState().armHints({ itemId: targetId })
          })
        })
        .finally(() => {
          setImporting(false)
          useWorkflowsStore.getState().setImporting(null)
        })
    },
    [centre, importing, items],
  )

  return { importWorkflow }
}
