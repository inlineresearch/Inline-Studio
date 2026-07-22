/** Live host + GPU telemetry strip for the Trainer tab, fed by `onSystemStats`. */
import { useTrainingStore } from '../../store/trainingStore'

function gb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

function Meter({
  label,
  fraction,
  detail,
}: {
  label: string
  fraction: number
  detail: string
}): React.JSX.Element {
  return (
    <div className="flex min-w-[120px] flex-1 flex-col gap-1">
      <div className="flex justify-between text-[11px] text-zinc-400">
        <span>{label}</span>
        <span className="tabular-nums text-zinc-500">{detail}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-emerald-500 transition-[width] duration-500"
          style={{ width: `${Math.min(100, Math.max(0, fraction * 100))}%` }}
        />
      </div>
    </div>
  )
}

export function SystemStatsBar(): React.JSX.Element {
  const stats = useTrainingStore((s) => s.systemStats)
  if (!stats) {
    return (
      <div className="border-t border-border bg-surface/60 px-4 py-2 text-[11px] text-zinc-500">
        Waiting for hardware stats…
      </div>
    )
  }
  return (
    <div className="flex flex-wrap items-center gap-4 border-t border-border bg-surface/60 px-4 py-2">
      <Meter label="CPU" fraction={stats.cpu / 100} detail={`${stats.cpu.toFixed(0)}%`} />
      <Meter
        label="RAM"
        fraction={stats.ramTotal ? stats.ramUsed / stats.ramTotal : 0}
        detail={`${gb(stats.ramUsed)} / ${gb(stats.ramTotal)}`}
      />
      {stats.gpus.map((g) => (
        <Meter
          key={g.index}
          label={`GPU ${g.index} · ${g.name}`}
          fraction={g.memoryTotal ? g.memoryUsed / g.memoryTotal : 0}
          detail={`${g.utilization.toFixed(0)}% · ${gb(g.memoryUsed)}/${gb(g.memoryTotal)}${
            g.temperature != null ? ` · ${g.temperature.toFixed(0)}°C` : ''
          }`}
        />
      ))}
    </div>
  )
}
