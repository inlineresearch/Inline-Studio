/**
 * The Trainer sidebar's Outputs tab: what this project's training has produced.
 *
 * Run-centric rather than a file listing, because a run is what yields both artifacts - a finished
 * LoRA, or a checkpoint you can resume. (The global `models/loras/` catalog is a different thing;
 * the LoRA loader node's dropdown already covers that.)
 */
import type { TrainingRun } from '@shared/types'
import { studio } from '@/lib/studio'
import { useTrainingStore } from '../../store/trainingStore'
import { PlayIcon } from '../../components/icons'

function when(ms: number): string {
  const d = new Date(ms)
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

/** The filename a run produced, without the `loras/` prefix. */
function loraName(run: TrainingRun): string {
  return (run.outputLoraPath ?? '').split('/').pop() ?? ''
}

function DoneRow({ run }: { run: TrainingRun }): React.JSX.Element {
  const hp = run.hyperparams
  return (
    <div className="flex flex-col gap-1 border-b border-border px-3 py-2">
      <span className="truncate text-sm text-zinc-100" title={loraName(run)}>
        {loraName(run)}
      </span>
      <span className="text-[10px] text-zinc-500">
        rank {hp.rank} · {run.totalSteps} steps · {hp.resolution}px · {when(run.updatedAt)}
      </span>
      <div className="flex gap-2 pt-0.5">
        <button
          onClick={() => void studio().clipboard.writeText(run.outputLoraPath ?? '')}
          className="rounded border border-border px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-panel"
        >
          Copy path
        </button>
      </div>
    </div>
  )
}

function ResumableRow({ run }: { run: TrainingRun }): React.JSX.Element {
  const resume = useTrainingStore((s) => s.resume)
  const runs = useTrainingStore((s) => s.runs)
  const blocked = runs.some((r) => r.status === 'training' || r.status === 'queued')
  return (
    <div className="flex flex-col gap-1 border-b border-border px-3 py-2">
      <span className="truncate text-sm text-zinc-100">{run.name}</span>
      <span className="text-[10px] text-zinc-500">
        stopped at {run.step}/{run.totalSteps} · {when(run.updatedAt)}
      </span>
      {run.error && <span className="text-[10px] text-amber-400/80">{run.error}</span>}
      <div className="flex gap-2 pt-0.5">
        <button
          onClick={() => void resume(run.id)}
          disabled={blocked}
          title={blocked ? 'Another run is using the GPU' : 'Resume from the last checkpoint'}
          className="flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-panel disabled:opacity-40"
        >
          <PlayIcon className="h-3 w-3" /> Resume
        </button>
      </div>
    </div>
  )
}

export function OutputsPanel(): React.JSX.Element {
  const runs = useTrainingStore((s) => s.runs)
  const done = runs.filter((r) => r.status === 'done' && r.outputLoraPath)
  const resumable = runs.filter((r) => r.status === 'interrupted' || r.status === 'failed')

  if (done.length === 0 && resumable.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center p-6 text-center text-sm text-zinc-500">
        Trained LoRAs and resumable runs show up here.
      </div>
    )
  }
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      {done.length > 0 && (
        <>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
            LoRAs ({done.length})
          </div>
          {done.map((r) => (
            <DoneRow key={r.id} run={r} />
          ))}
        </>
      )}
      {resumable.length > 0 && (
        <>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
            Resumable ({resumable.length})
          </div>
          {resumable.map((r) => (
            <ResumableRow key={r.id} run={r} />
          ))}
        </>
      )}
    </div>
  )
}
