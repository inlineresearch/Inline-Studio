/**
 * What a new project shows instead of an empty canvas: four ways to start, each saying how it will
 * run on this machine, each one click away from a working graph.
 *
 * The overlay stays `pointer-events-none` so dragging an asset onto an empty canvas still reaches
 * the ReactFlow pane underneath; only the card grid takes pointer events.
 */
import { useEffect, useMemo } from 'react'

import { STARTER_RECIPES } from '../../../lib/starterRecipes'
import { pickRecommended, readVram, recommendedLabel, tierFor } from '../../../lib/vramAdvice'
import type { StarterKey } from '../../../lib/vramAdvice'
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import { useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { useTrainingStore } from '../../../store/trainingStore'
import { BoxIcon, WandIcon } from '../nodes/NodeBadge'
import { StarterCard, type CardStatus } from './StarterCard'

const ICONS: Record<StarterKey, React.JSX.Element> = {
  zimage: <WandIcon className="h-4 w-4" />,
  flux2: <WandIcon className="h-4 w-4" />,
  krea2: <WandIcon className="h-4 w-4" />,
  training: <BoxIcon className="h-4 w-4" />,
}

export function StarterCards({ onPick }: { onPick: (key: StarterKey) => void }): React.JSX.Element {
  const stats = useTrainingStore((s) => s.systemStats)
  const byType = useModelRequirementsStore((s) => s.byType)
  const loadReqs = useModelRequirementsStore((s) => s.load)
  const openReqs = useModelRequirementsStore((s) => s.open)
  const descriptors = useCoreNodesStore((s) => s.descriptors)
  const registryVersion = useCoreNodesStore((s) => s.registryVersion)

  const reading = useMemo(() => readVram(stats), [stats])
  const recommended = useMemo(() => pickRecommended(reading), [reading])
  const label = recommendedLabel(reading)

  // Requirements are a filesystem stat in Core, so this is cheap and needs no node on the canvas.
  // Re-run on registryVersion: a dropped-in file or a finished download bumps it.
  useEffect(() => {
    for (const recipe of STARTER_RECIPES) if (recipe.coreType) void loadReqs(recipe.coreType)
  }, [loadReqs, registryVersion])

  const known = useMemo(() => new Set(descriptors.map((d) => d.type)), [descriptors])
  // Core may not have answered yet. Show the cards muted rather than letting the layout jump.
  const engineReady = descriptors.length > 0

  const statusFor = (coreType: string | null): CardStatus | null => {
    if (!coreType) return null
    const reqs = byType[coreType]
    if (!reqs) return null
    const missing = reqs.components.filter((c) => !c.present && !c.optional).length
    // The engine's estimate sizes what is on disk, so it only means anything once installed.
    const note = reqs.allPresent ? (reqs.estimate?.warning ?? null) : null
    return { missing, note }
  }

  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center p-6">
      <div className="pointer-events-auto flex w-full max-w-3xl flex-col gap-3">
        <div className="text-center">
          <p className="text-sm font-medium text-zinc-200">Start with a model</p>
          <p className="mt-1 text-xs text-zinc-500">
            {reading.state === 'pending'
              ? 'Checking your hardware'
              : reading.state === 'unknown'
                ? 'Could not read GPU memory, so these show their usual requirements.'
                : `Matched to your ${reading.name}, ${reading.totalGb.toFixed(0)} GB.`}
          </p>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {STARTER_RECIPES.map((recipe) => {
            const unavailable =
              recipe.coreType != null && engineReady && !known.has(recipe.coreType)
            if (unavailable) return null
            return (
              <StarterCard
                key={recipe.key}
                icon={ICONS[recipe.key]}
                title={recipe.title}
                blurb={recipe.blurb}
                advice={tierFor(recipe.key, reading)}
                recommended={recipe.key === recommended}
                recommendedLabel={label}
                status={statusFor(recipe.coreType)}
                onPick={() => onPick(recipe.key)}
                onGetModels={
                  recipe.coreType ? () => openReqs(recipe.coreType as string) : undefined
                }
                disabled={recipe.coreType != null && !engineReady}
              />
            )
          })}
        </div>
        <p className="text-center text-[10px] text-zinc-600">
          Or drag an asset from the Assets panel to start from an image.
        </p>
      </div>
    </div>
  )
}
