import { Handle, Position, type NodeProps } from '@xyflow/react'
import { portKindColor } from '@shared/coreNodes'
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import { useGenerationStore } from '../../../store/generationStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { NodeFrame } from './NodeFrame'
import {
  AdjustIcon,
  BoxIcon,
  ImageGlyph,
  NodeBadge,
  NodeBadgeRow,
  PlayIcon,
  SquareIcon,
  TypeIcon,
  WandIcon,
} from './NodeBadge'
import { resolveMedia } from '@/lib/media'

interface GraphNodeData extends Record<string, unknown> {
  itemId: string
}

/** Even vertical spacing for `count` handles down an edge. */
function edgePercent(index: number, count: number): string {
  return `${(((index + 1) / (count + 1)) * 100).toFixed(2)}%`
}

/** Map a Core descriptor's `icon` string to a node-family glyph (falls back to the square). */
function coreGlyph(icon: string): React.JSX.Element {
  switch (icon) {
    case 'wand':
      return <WandIcon />
    case 'box':
      return <BoxIcon />
    case 'type':
      return <TypeIcon />
    case 'image':
      return <ImageGlyph />
    default:
      return <SquareIcon />
  }
}

/**
 * A generic Inline Core graph node backed by a `core` moodboard item. Resolves its descriptor from
 * the served `/v1/models` palette and renders in the same card style as the fal Generate node: a
 * floating title badge, an edge-to-edge output preview, and a footer with Run + an adjust (settings)
 * button. Params live behind the adjust button in the Core settings sidebar — the node face stays
 * clean, so a model node like Z-Image Turbo reads as one simple node. One colored handle per
 * declared port (inputs left, outputs right, colored by kind).
 */
export function GraphNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { itemId } = data as GraphNodeData
  const item = useMoodboardStore((s) => s.items.find((i) => i.id === itemId))
  const coreType = item?.type === 'core' ? item.data.core?.type : undefined
  const descriptor = useCoreNodesStore((s) =>
    coreType ? s.descriptors.find((d) => d.type === coreType) : undefined,
  )
  const runWorkflow = useGenerationStore((s) => s.runWorkflow)
  const toggleSettings = useGenerationStore((s) => s.toggleCoreSettings)
  const busy = useGenerationStore((s) => s.busyByFrame[itemId] ?? false)
  const progress = useGenerationStore((s) => s.progressByFrame[itemId])
  const status = useGenerationStore((s) => s.statusByFrame[itemId])

  if (!item || item.type !== 'core' || !item.data.core || !descriptor) {
    return (
      <NodeFrame id={id} selected={!!selected} minWidth={200} minHeight={92} subtleSelect>
        <div className="flex h-full flex-col items-center justify-center gap-1 p-3 text-center">
          {coreType ? (
            <>
              <span className="text-[11px] font-semibold text-amber-300">Node unavailable</span>
              <span className="text-[10px] leading-tight text-zinc-400">
                <span className="text-zinc-300">{coreType}</span> is not registered. Start Inline
                Core and install its runtime (the <span className="text-zinc-300">zimage</span>{' '}
                extra).
              </span>
            </>
          ) : (
            <span className="text-[11px] text-zinc-500">Core node</span>
          )}
        </div>
      </NodeFrame>
    )
  }

  const core = item.data.core
  const missingCategories = descriptor.params
    .filter((p) => p.optionsFrom && (p.options?.length ?? 0) === 0)
    .map((p) => p.optionsFrom as string)
  const pct = typeof progress === 'number' ? Math.round(progress * 100) : null

  return (
    <>
      {/* Floating title badge — matches the fal Generate node. */}
      <NodeBadgeRow>
        <NodeBadge icon={coreGlyph(descriptor.icon)}>{descriptor.title}</NodeBadge>
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
          {/* Edge-to-edge output preview. */}
          <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
            {core.output?.kind === 'image' ? (
              <img
                src={resolveMedia(core.output.filePath)}
                alt=""
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center px-4">
                <span className="text-center text-[10px] text-zinc-600">
                  {busy
                    ? (status ?? 'Working…')
                    : missingCategories.length > 0
                      ? `Model files not found (${missingCategories.join(', ')})`
                      : 'Run to generate'}
                </span>
              </div>
            )}

            {busy && (
              <>
                <span className="absolute left-2 top-2 flex max-w-[calc(100%-1rem)] items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 text-[10px] font-medium text-emerald-300 backdrop-blur">
                  <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-emerald-400" />
                  <span className="truncate">
                    {status ?? (pct != null ? `${pct}%` : 'Working…')}
                  </span>
                </span>
                <div className="absolute inset-x-0 bottom-0 h-1 bg-black/40">
                  <div
                    className="h-full bg-emerald-400 transition-all"
                    style={{ width: `${pct ?? 0}%` }}
                  />
                </div>
              </>
            )}
          </div>

          {/* Footer: category label + run + settings (adjust) — same layout as the fal node. */}
          <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border bg-surface/90 px-1.5 py-1">
            <span className="truncate px-1 text-[10px] uppercase tracking-wide text-zinc-500">
              {descriptor.category}
            </span>
            <div className="flex shrink-0 items-center gap-0.5">
              <button
                onClick={() => void runWorkflow(itemId)}
                disabled={busy}
                title="Run up to this node"
                className="nodrag flex h-6 w-6 items-center justify-center rounded text-emerald-400 hover:bg-black/40 hover:text-emerald-300 disabled:opacity-40"
              >
                <PlayIcon />
              </button>
              <button
                onClick={() => toggleSettings(itemId)}
                title="Settings"
                data-gen-settings-toggle
                className="nodrag flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
              >
                <AdjustIcon />
              </button>
            </div>
          </div>
        </div>
      </NodeFrame>

      {descriptor.inputs.map((port, i) => (
        <Handle
          key={port.id}
          type="target"
          id={port.id}
          position={Position.Left}
          style={{
            top: edgePercent(i, descriptor.inputs.length),
            background: portKindColor(port.kind),
          }}
          className="group !h-3 !w-3 !border-2 !border-surface"
        >
          <span className="pointer-events-none absolute right-full top-1/2 z-50 mr-2 hidden -translate-y-1/2 whitespace-nowrap rounded bg-black/90 px-1.5 py-0.5 text-[10px] leading-none text-zinc-100 shadow group-hover:block">
            {port.label} <span className="text-zinc-400">· {port.kind}</span>
          </span>
        </Handle>
      ))}
      {descriptor.outputs.map((port, i) => (
        <Handle
          key={port.id}
          type="source"
          id={port.id}
          position={Position.Right}
          style={{
            top: edgePercent(i, descriptor.outputs.length),
            background: portKindColor(port.kind),
          }}
          className="group !h-3 !w-3 !border-2 !border-surface"
        >
          <span className="pointer-events-none absolute left-full top-1/2 z-50 ml-2 hidden -translate-y-1/2 whitespace-nowrap rounded bg-black/90 px-1.5 py-0.5 text-[10px] leading-none text-zinc-100 shadow group-hover:block">
            {port.label} <span className="text-zinc-400">· {port.kind}</span>
          </span>
        </Handle>
      ))}
    </>
  )
}
