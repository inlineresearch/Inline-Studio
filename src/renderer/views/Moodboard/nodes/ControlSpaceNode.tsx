import { useEffect } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { resolveMedia } from '@/lib/media'
import { useAssetStore } from '../../../store/assetStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useControlSpaceStore } from '../../../store/controlSpaceStore'
import { useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { NodeFrame } from './NodeFrame'
import { BoxIcon, DownloadIcon, NodeBadge, NodeBadgeRow } from './NodeBadge'

// Reuse the shared requirements/download flow under this pseudo node type (a Core provider is
// registered for it), so the global popup + progress events cover Control Space too.
const REQ_TYPE = 'controlSpace'

/**
 * "Control Space": a source node whose face shows the last rendered OpenPose control map and opens
 * the full-screen 3D pose editor. Its render (a library asset in `data.controlAssetId`) feeds a gen
 * node's control input via the output handle. When no ControlNet model is on disk it offers a
 * one-click download (the map needs a ControlNet downstream to actually steer generation).
 */
export function ControlSpaceNode({ id, selected }: NodeProps): React.JSX.Element {
  const item = useMoodboardStore((s) => s.items.find((it) => it.id === id))
  const assets = useAssetStore((s) => s.assets)
  const openEditor = useControlSpaceStore((s) => s.open)
  const loadReqs = useModelRequirementsStore((s) => s.load)
  const openReqs = useModelRequirementsStore((s) => s.open)
  const reqs = useModelRequirementsStore((s) => s.byType[REQ_TYPE])
  const suggestedDl = useModelRequirementsStore((s) => s.downloads[REQ_TYPE]?.controlnet)

  useEffect(() => {
    void loadReqs(REQ_TYPE)
  }, [loadReqs])

  const assetId = item?.data.controlAssetId
  const asset = assetId ? assets.find((a) => a.id === assetId) : undefined
  const src = asset ? resolveMedia(asset.filePath) : null
  const scene = item?.data.controlScene
  const hint = (scene?.applyPromptHint ?? true) ? (scene?.promptHint ?? null) : null
  const suggested = reqs?.components.find((c) => c.optional && !c.present) ?? null
  const suggestedPct = suggestedDl ? Math.round(suggestedDl.fraction * 100) : null

  return (
    <>
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<BoxIcon />} title="Control Space">
          Control Space
        </NodeBadge>
        {suggested && (
          <button
            onClick={() => openReqs(REQ_TYPE)}
            title={`${suggested.label} - optional download so this map can steer a ControlNet gen`}
            className="nodrag flex h-6 items-center gap-1 rounded-full border border-border bg-panel/80 px-2 text-[10px] font-medium text-zinc-300 shadow-sm backdrop-blur hover:border-emerald-500/40 hover:text-emerald-300"
          >
            <DownloadIcon className="h-3.5 w-3.5" />
            {suggestedDl && !suggestedDl.error ? `ControlNet ${suggestedPct}%` : 'Get ControlNet'}
          </button>
        )}
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={200}
        padded={false}
        subtleSelect
      >
        <div className="relative flex h-full w-full flex-col">
          <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-black">
            {src ? (
              <img src={src} alt="Pose control map" className="h-full w-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-2 p-3 text-center">
                <span className="text-[11px] text-zinc-500">
                  Pose a 3D character, then render an OpenPose control map.
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between gap-2 border-t border-border bg-surface/90 px-2 py-1">
            {/* The facing text this node adds to a downstream prompt - the control map itself has no
                way to say "facing away", so it is worth surfacing where the wiring is visible. */}
            <span
              className="truncate text-[10px] text-zinc-500"
              title={hint ? `Adds to the prompt: “${hint.positive}”` : undefined}
            >
              {hint ? '+ facing in prompt' : ''}
            </span>
            <button
              onClick={() => openEditor(id)}
              title="Open the 3D pose editor"
              className="nodrag flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-200 hover:bg-black/40 hover:text-white"
            >
              <BoxIcon className="h-3 w-3" />
              {src ? 'Edit pose' : 'Open Control Space'}
            </button>
          </div>
        </div>
      </NodeFrame>

      <Handle
        type="source"
        id="out"
        position={Position.Right}
        title="Control map"
        className="group !h-3 !w-3 !border-2 !border-surface !bg-pink-300"
      >
        <span className="pointer-events-none absolute left-full top-1/2 ml-1.5 hidden -translate-y-1/2 items-center whitespace-nowrap rounded-md border border-border bg-panel/95 px-1.5 py-0.5 text-[10px] font-medium text-zinc-200 shadow-sm backdrop-blur group-hover:flex">
          Control map
        </span>
      </Handle>
    </>
  )
}
