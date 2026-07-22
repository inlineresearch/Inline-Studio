/** A training run's live status: progress bar, a hand-rolled SVG loss curve, and sample previews. */
import type { TrainingRun } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { useTrainingStore } from '../../store/trainingStore'
import { PlayIcon } from '../../components/icons'

/** A minimal inline SVG loss sparkline (no chart lib, matching DirectorNode's hand-rolled visuals). */
function LossCurve({ loss }: { loss: number[] }): React.JSX.Element {
  if (loss.length < 2) {
    return (
      <div className="flex h-24 items-center justify-center text-[11px] text-zinc-600">
        No loss yet
      </div>
    )
  }
  const w = 100
  const h = 40
  const max = Math.max(...loss)
  const min = Math.min(...loss)
  const span = max - min || 1
  const points = loss
    .map((v, i) => {
      const x = (i / (loss.length - 1)) * w
      const y = h - ((v - min) / span) * h
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
  return (
    <div className="rounded-md border border-border bg-black/30 p-2">
      <svg viewBox={`0 0 ${w} ${h}`} className="h-24 w-full" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke="rgb(16 185 129)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-zinc-500">
        <span>min {min.toFixed(4)}</span>
        <span>latest {loss[loss.length - 1]!.toFixed(4)}</span>
      </div>
    </div>
  )
}

export function TrainingMonitor({ run }: { run: TrainingRun }): React.JSX.Element {
  const progress = useTrainingStore((s) => s.progressByRun[run.id])
  // `?? []` outside the selector: a fresh [] returned from a selector is a new ref every call, which
  // Zustand's Object.is equality reads as a change -> infinite re-render (React #185).
  const loss = useTrainingStore((s) => s.lossByRun[run.id]) ?? []
  const samples = useTrainingStore((s) => s.samplesByRun[run.id]) ?? []
  const cancel = useTrainingStore((s) => s.cancel)
  const resume = useTrainingStore((s) => s.resume)

  const fraction = progress?.fraction ?? run.progressFraction
  const step = progress?.step ?? run.step
  const total = progress?.totalSteps || run.totalSteps
  const active = run.status === 'training' || run.status === 'queued'

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface/70 p-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-sm font-medium text-zinc-100">{run.name}</span>
          <span className="text-[11px] text-zinc-500">
            {run.status}
            {progress?.status ? ` · ${progress.status}` : ''} · step {step}/{total || '?'}
          </span>
        </div>
        {active ? (
          <button
            onClick={() => void cancel(run.id)}
            className="rounded-md border border-border px-2 py-1 text-[11px] text-zinc-300 hover:bg-panel"
          >
            Cancel
          </button>
        ) : run.status === 'interrupted' || run.status === 'failed' ? (
          <button
            onClick={() => void resume(run.id)}
            className="flex items-center gap-1 rounded-md bg-emerald-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-emerald-500"
          >
            <PlayIcon className="h-3 w-3" /> Resume
          </button>
        ) : null}
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-emerald-500 transition-[width] duration-300"
          style={{ width: `${Math.min(100, Math.max(0, fraction * 100))}%` }}
        />
      </div>

      {run.error && <div className="text-[11px] text-red-400">{run.error}</div>}
      {run.status === 'done' && run.outputLoraPath && (
        <div className="text-[11px] text-emerald-400">
          Saved {run.outputLoraPath.split('/').pop()} — pick it in a LoRA loader node.
        </div>
      )}

      <LossCurve loss={loss} />

      {samples.length > 0 && (
        <div className="flex gap-2 overflow-x-auto">
          {samples.slice(-6).map((path) => (
            <img
              key={path}
              src={resolveMedia(path)}
              alt="training sample"
              className="h-20 w-20 shrink-0 rounded-md object-cover"
            />
          ))}
        </div>
      )}
    </div>
  )
}
