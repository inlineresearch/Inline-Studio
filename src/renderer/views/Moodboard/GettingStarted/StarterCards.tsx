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
import { useFalSettingsStore } from '../../../store/falSettingsStore'
import { useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { useTrainingStore } from '../../../store/trainingStore'
import { useUiStore } from '../../../store/uiStore'
import { LayersIcon, VideoGlyph, WandIcon } from '../nodes/NodeBadge'
import { StarterCard, type CardStatus } from './StarterCard'

// The glyph reinforces the kind colour: a wand for image generation, a camera for video, a stack
// for the dataset that training starts from.
const ICONS: Record<StarterKey, React.JSX.Element> = {
  minimaxh3: <VideoGlyph className="h-4 w-4" />,
  zimage: <WandIcon className="h-4 w-4" />,
  flux2: <WandIcon className="h-4 w-4" />,
  krea2: <WandIcon className="h-4 w-4" />,
  training: <LayersIcon className="h-4 w-4" />,
}

export function StarterCards({ onPick }: { onPick: (key: StarterKey) => void }): React.JSX.Element {
  const stats = useTrainingStore((s) => s.systemStats)
  const byType = useModelRequirementsStore((s) => s.byType)
  const loadReqs = useModelRequirementsStore((s) => s.load)
  const openReqs = useModelRequirementsStore((s) => s.open)
  const descriptors = useCoreNodesStore((s) => s.descriptors)
  const registryVersion = useCoreNodesStore((s) => s.registryVersion)
  const falConfigured = useFalSettingsStore((s) => s.configured)
  const loadFalStatus = useFalSettingsStore((s) => s.load)
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen)

  useEffect(() => void loadFalStatus(), [loadFalStatus])

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
      <div className="pointer-events-auto flex w-full max-w-2xl flex-col gap-3">
        <div className="text-center">
          <p className="text-base font-semibold text-zinc-50">Start with a model</p>
          <p className="mt-1 text-xs text-zinc-400">
            {reading.state === 'pending'
              ? 'Checking your hardware'
              : reading.state === 'unknown'
                ? 'Could not read GPU memory, so these show their usual requirements.'
                : `Matched to your ${reading.name}, ${reading.totalGb.toFixed(0)} GB.`}
          </p>
        </div>
        <div className="flex flex-col gap-2">
          {STARTER_RECIPES.map((recipe) => {
            const unavailable =
              recipe.coreType != null && engineReady && !known.has(recipe.coreType)
            if (unavailable) return null
            return (
              <StarterCard
                key={recipe.key}
                icon={ICONS[recipe.key]}
                title={recipe.title}
                tag={recipe.tag}
                kind={recipe.kind}
                blurb={recipe.blurb}
                advice={tierFor(recipe.key, reading)}
                recommended={recipe.key === recommended}
                recommendedLabel={label}
                status={statusFor(recipe.coreType)}
                action={
                  recipe.falModelId && !falConfigured
                    ? {
                        hint: 'Needs a fal API key',
                        label: 'Add key',
                        onClick: () => setSettingsOpen(true),
                      }
                    : undefined
                }
                onPick={() => onPick(recipe.key)}
                onGetModels={
                  recipe.coreType ? () => openReqs(recipe.coreType as string) : undefined
                }
                disabled={recipe.coreType != null && !engineReady}
              />
            )
          })}
        </div>
        <p className="text-center text-[11px] text-zinc-500">
          Or drag an asset from the Assets panel to start from an image.
        </p>
      </div>
    </div>
  )
}
