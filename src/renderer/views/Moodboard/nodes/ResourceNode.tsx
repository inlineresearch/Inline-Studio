/**
 * Utility node: live host telemetry (CPU / RAM / VRAM) as circular gauges.
 *
 * Deliberately connection-less - it reads the machine, it doesn't take part in the graph - so it
 * carries no handles. Usable on either canvas (Studio and Trainer), fed by the `onSystemStats`
 * broadcast the Trainer store already subscribes to.
 */
import type { NodeProps } from '@xyflow/react'
import { useTrainingStore } from '../../../store/trainingStore'
import { NodeFrame } from './NodeFrame'
import { CpuIcon, NodeBadge, NodeBadgeRow } from './NodeBadge'

function gb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)}`
}

/** A ring gauge: an SVG circle whose stroke-dash sweeps with `fraction`. */
function Gauge({
  label,
  fraction,
  center,
  detail,
}: {
  label: string
  fraction: number
  center: string
  detail: string
}): React.JSX.Element {
  const pct = Math.min(1, Math.max(0, Number.isFinite(fraction) ? fraction : 0))
  const r = 22
  const circumference = 2 * Math.PI * r
  // Amber past 75%, red past 90% - a glanceable "am I about to OOM" signal.
  const stroke = pct > 0.9 ? 'text-red-400' : pct > 0.75 ? 'text-amber-400' : 'text-emerald-400'
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center gap-1">
      <div className="relative h-[58px] w-[58px]">
        <svg viewBox="0 0 58 58" className="h-full w-full -rotate-90">
          <circle cx="29" cy="29" r={r} fill="none" strokeWidth="5" className="stroke-zinc-800" />
          <circle
            cx="29"
            cy="29"
            r={r}
            fill="none"
            strokeWidth="5"
            strokeLinecap="round"
            className={`${stroke} transition-[stroke-dashoffset] duration-500`}
            stroke="currentColor"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - pct)}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-[11px] font-medium tabular-nums text-zinc-200">
          {center}
        </span>
      </div>
      <span className="text-[10px] font-medium text-zinc-400">{label}</span>
      <span className="truncate text-[9px] tabular-nums text-zinc-500">{detail}</span>
    </div>
  )
}

export function ResourceNode({ id, selected }: NodeProps): React.JSX.Element {
  const stats = useTrainingStore((s) => s.systemStats)
  // One GPU per node keeps the card readable; index 0 is the training GPU on a single-GPU box.
  const gpu = stats?.gpus[0]

  return (
    <>
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<CpuIcon />}>Resources</NodeBadge>
        {gpu && (
          <NodeBadge tone="info" title={gpu.name}>
            {gpu.name}
          </NodeBadge>
        )}
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect minWidth={220}>
        <div className="flex h-full flex-col">
          <div className="flex flex-1 items-center justify-around gap-2 px-3 py-3">
            {stats ? (
              <>
                <Gauge
                  label="CPU"
                  fraction={stats.cpu / 100}
                  center={`${stats.cpu.toFixed(0)}%`}
                  detail="utilisation"
                />
                <Gauge
                  label="RAM"
                  fraction={stats.ramTotal ? stats.ramUsed / stats.ramTotal : 0}
                  center={`${gb(stats.ramUsed)}G`}
                  detail={`of ${gb(stats.ramTotal)} GB`}
                />
                <Gauge
                  label="VRAM"
                  fraction={gpu && gpu.memoryTotal ? gpu.memoryUsed / gpu.memoryTotal : 0}
                  center={gpu ? `${gb(gpu.memoryUsed)}G` : '—'}
                  detail={gpu ? `of ${gb(gpu.memoryTotal)} GB` : 'no GPU'}
                />
              </>
            ) : (
              <span className="text-[11px] text-zinc-500">Waiting for hardware stats…</span>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-border bg-surface/90 px-2 py-1">
            <span className="truncate text-[10px] text-zinc-500">
              {gpu ? `GPU ${gpu.index} · ${gpu.utilization.toFixed(0)}% busy` : 'Host'}
            </span>
            <span className="text-[10px] tabular-nums text-zinc-500">
              {gpu?.temperature != null ? `${gpu.temperature.toFixed(0)}°C` : ''}
            </span>
          </div>
        </div>
      </NodeFrame>
    </>
  )
}
