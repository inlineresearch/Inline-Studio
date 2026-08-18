/**
 * Graph node: the loss curve for a wired training run. A hand-rolled SVG sparkline (no chart lib),
 * matching how DirectorNode hand-rolls its timeline.
 */
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTrainingStore } from '../../../store/trainingStore'
import { NodeFrame } from '../../Moodboard/nodes/NodeFrame'
import { ChartIcon, NodeBadge, NodeBadgeRow } from '../../Moodboard/nodes/NodeBadge'
import { RUN_HANDLE } from './handles'
import { useBoardActions } from '../../Moodboard/nodes/boardActions'

/** Resolve the run id from an incoming edge (Trainer → Graph). */
function wiredRunId(
  itemId: string,
  connectors: { fromItemId: string; toItemId: string }[],
  items: { id: string; data: { runId?: string | null } }[],
): string | null {
  const incoming = connectors.find((c) => c.toItemId === itemId)
  if (!incoming) return null
  return items.find((i) => i.id === incoming.fromItemId)?.data.runId ?? null
}

/** Compact axis tick: 0.0234 style, but falls back to exponent form for very small losses. */
function tick(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(1)
  return v.toFixed(4)
}

function Curve({ loss, lastStep }: { loss: number[]; lastStep: number }): React.JSX.Element {
  if (loss.length < 2) {
    return (
      <div className="flex h-full items-center justify-center text-[11px] text-zinc-600">
        Waiting for loss…
      </div>
    )
  }
  const w = 100
  const h = 40
  const min = Math.min(...loss)
  const max = Math.max(...loss)
  const span = max - min || 1
  const mid = (min + max) / 2
  const points = loss
    .map((v, i) => `${(i / (loss.length - 1)) * w},${h - ((v - min) / span) * h}`)
    .join(' ')
  // Axis labels are HTML around the SVG, not inside it: the plot uses
  // preserveAspectRatio="none" to fill the node, which would distort any text drawn in it.
  return (
    <div className="flex h-full w-full flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="flex w-11 shrink-0 flex-col justify-between py-px pr-1 text-right text-[8px] tabular-nums leading-none text-zinc-500">
          <span>{tick(max)}</span>
          <span>{tick(mid)}</span>
          <span>{tick(min)}</span>
        </div>
        <div className="relative min-w-0 flex-1 border-b border-l border-zinc-700">
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="h-full w-full">
            <line
              x1="0"
              y1={h / 2}
              x2={w}
              y2={h / 2}
              className="stroke-zinc-800"
              strokeWidth="0.5"
              vectorEffect="non-scaling-stroke"
            />
            <polyline
              points={points}
              fill="none"
              stroke="currentColor"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
              className="text-emerald-400"
            />
          </svg>
        </div>
      </div>
      <div className="flex justify-between pl-11 pt-0.5 text-[8px] tabular-nums leading-none text-zinc-500">
        <span>0</span>
        <span>step {lastStep}</span>
      </div>
    </div>
  )
}

export function LossGraphNode({ id, selected }: NodeProps): React.JSX.Element {
  const { items, connectors } = useBoardActions()
  const runId = wiredRunId(id, connectors, items)
  const loss = useTrainingStore((s) => (runId ? s.lossByRun[runId] : undefined)) ?? []
  const progress = useTrainingStore((s) => (runId ? s.progressByRun[runId] : undefined))
  const latest = loss.length ? loss[loss.length - 1] : null
  // One loss point per progress tick, so the point count is the step when the run reports none yet.
  const lastStep = progress?.step ?? loss.length

  return (
    <>
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<ChartIcon />}>Graph</NodeBadge>
        {latest != null && (
          <NodeBadge tone="info" accent="text-emerald-400">
            {latest.toFixed(4)}
          </NodeBadge>
        )}
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect minWidth={200}>
        <div className="flex h-full flex-col">
          <div className="flex-1 bg-black px-2 py-2">
            {runId ? (
              <Curve loss={loss} lastStep={lastStep} />
            ) : (
              <div className="flex h-full items-center justify-center text-[11px] text-zinc-600">
                Wire a Train LoRA node
              </div>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-border bg-surface/90 px-2 py-1">
            <span className="text-[10px] text-zinc-500">loss</span>
            <span className="text-[10px] tabular-nums text-zinc-500">{loss.length} pts</span>
          </div>
        </div>
      </NodeFrame>
      <Handle
        type="target"
        position={Position.Left}
        id={RUN_HANDLE}
        className="group !h-3 !w-3 !border-2 !border-surface !bg-violet-400"
        title="Run"
      />
    </>
  )
}
