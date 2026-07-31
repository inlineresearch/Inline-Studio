/**
 * Turns a starter-card click into a built, selected, framed graph with the hints armed.
 *
 * The selection step is not cosmetic. The Run pill only mounts while its node is the selected
 * graph's run target, and `toNodes` rebuilds every node without `selected` whenever `items`
 * changes, so selecting has to happen *after* the new items land. Hence the pending ref plus an
 * effect keyed on items rather than a straight call.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Node } from '@xyflow/react'

import { buildStarterGraph } from '../../../lib/starterGraph'
import { recipeFor } from '../../../lib/starterRecipes'
import type { StarterKey } from '../../../lib/vramAdvice'
import { useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useOnboardingStore } from '../../../store/onboardingStore'
import { useUiStore } from '../../../store/uiStore'

interface Options {
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>
  fitView: (opts: {
    nodes: { id: string }[]
    padding: number
    maxZoom: number
    duration: number
  }) => void
  centre: () => { x: number; y: number }
}

export function useStarterGraph({ setNodes, fitView, centre }: Options): {
  onPick: (key: StarterKey) => void
  building: boolean
} {
  const items = useMoodboardStore((s) => s.items)
  const [building, setBuilding] = useState(false)
  // Ids waiting to be selected once they appear in `items`.
  const pending = useRef<string[] | null>(null)
  // Node the hints should point at, held back until its models are installed.
  const deferred = useRef<{ itemId: string; coreType: string } | null>(null)

  useEffect(() => {
    const ids = pending.current
    if (!ids || !ids.every((id) => items.some((i) => i.id === id))) return
    pending.current = null
    setNodes((nodes) => nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })))
    fitView({ nodes: ids.map((id) => ({ id })), padding: 0.35, maxZoom: 1, duration: 400 })
  }, [items, setNodes, fitView])

  // Models can take a long time to download, and a Run hint is useless until Run works. So the
  // hint waits for the requirements to report everything present.
  const byType = useModelRequirementsStore((s) => s.byType)
  useEffect(() => {
    const held = deferred.current
    if (!held) return
    if (!byType[held.coreType]?.allPresent) return
    deferred.current = null
    useOnboardingStore.getState().armHints({ itemId: held.itemId, surface: 'studio' })
  }, [byType])

  const onPick = useCallback(
    (key: StarterKey): void => {
      if (building) return
      const recipe = recipeFor(key)
      if (!recipe) return

      // Training lives on its own canvas, which seeds its own graph when empty.
      if (!recipe.coreType) {
        useUiStore.getState().setActiveTab('trainer')
        return
      }

      setBuilding(true)
      const coreType = recipe.coreType
      void buildStarterGraph(recipe, centre())
        .then((ids) => {
          if (ids.length !== 2) return
          pending.current = ids
          const genId = ids[1]
          const ready = useModelRequirementsStore.getState().byType[coreType]?.allPresent
          if (ready) {
            useOnboardingStore.getState().armHints({ itemId: genId, surface: 'studio' })
          } else {
            // The graph is the durable artifact, so it is built either way; the download popup
            // opens on top and the hints wait behind it.
            deferred.current = { itemId: genId, coreType }
            useModelRequirementsStore.getState().open(coreType)
          }
        })
        .finally(() => setBuilding(false))
    },
    [building, centre],
  )

  return { onPick, building }
}
