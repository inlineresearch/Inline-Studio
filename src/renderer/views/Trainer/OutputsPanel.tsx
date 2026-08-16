/**
 * The Trainer sidebar's Outputs tab: what this project's training has produced.
 *
 * Run-centric rather than a file listing, because a run is what yields both artifacts - a finished
 * LoRA, or a checkpoint you can resume. (The global `models/loras/` catalog is a different thing;
 * the LoRA loader node's dropdown already covers that.)
 */
import { useEffect, useState } from 'react'
import type { TrainingRun, TrainingSnapshot } from '@shared/types'
import { useTrainingStore } from '../../store/trainingStore'
import { ChevronDownIcon, ChevronRightIcon, DownloadIcon, PlayIcon } from '../../components/icons'

function when(ms: number): string {
  const d = new Date(ms)
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

/** The filename a run produced, without the `loras/` prefix. */
function loraName(run: TrainingRun): string {
  return (run.outputLoraPath ?? '').split('/').pop() ?? ''
}

/** The snapshot's filename on disk, e.g. `step-000500.safetensors`. */
function snapshotName(snap: TrainingSnapshot): string {
  return snap.path.split('/').pop() || `step-${snap.step}.safetensors`
}

function formatSize(bytes: number): string {
  return bytes >= 1024 ** 2
    ? `${Math.round(bytes / 1024 ** 2)} MB`
    : `${Math.round(bytes / 1024)} KB`
}

/**
 * One run, collapsed to a single line by default.
 *
 * A project accumulates runs and each one can carry a long list of mid-run snapshots, so expanded
 * cards would bury the run you are looking for. The snapshot count is loaded even while collapsed,
 * because it is the thing that tells you a card is worth opening.
 */
function RunCard({
  run,
  title,
  subtitle,
  action,
  note,
  children,
}: {
  run: TrainingRun
  title: string
  subtitle: string
  /** The run's own artifact, kept on the header: it is the reason the card exists. */
  action: React.ReactNode
  /** Why a run stopped. Stays visible collapsed, since it is the reason to look at the card. */
  note?: string | null
  /** Revealed on expand. The mid-run snapshots, which are the long, browsable part. */
  children: React.ReactNode
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const snapshots = useTrainingStore((s) => s.snapshotsByRun[run.id])
  const loadSnapshots = useTrainingStore((s) => s.loadSnapshots)
  const count = snapshots?.length ?? 0

  useEffect(() => {
    void loadSnapshots(run.id)
  }, [loadSnapshots, run.id, run.step])

  return (
    <div className="flex flex-col gap-1 border-b border-border px-3 py-2">
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={count === 0}
        aria-expanded={open}
        title={count === 0 ? 'No mid-run snapshots' : 'Show mid-run snapshots'}
        className="flex w-full items-center gap-1.5 text-left disabled:cursor-default"
      >
        {/* Only a card with something to reveal gets a chevron; the spacer keeps titles aligned. */}
        {count === 0 ? (
          <span className="h-3 w-3 shrink-0" />
        ) : open ? (
          <ChevronDownIcon className="h-3 w-3 shrink-0 text-zinc-500" />
        ) : (
          <ChevronRightIcon className="h-3 w-3 shrink-0 text-zinc-500" />
        )}
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm text-zinc-100" title={title}>
            {title}
          </span>
          <span className="block truncate text-[10px] text-zinc-500">{subtitle}</span>
        </span>
        {count > 0 && (
          <span
            title={`${count} mid-run snapshot${count === 1 ? '' : 's'}`}
            className="shrink-0 rounded bg-black/40 px-1.5 py-0.5 text-[10px] tabular-nums text-zinc-400"
          >
            {count}
          </span>
        )}
      </button>
      {note && <span className="text-[10px] text-amber-400/80">{note}</span>}
      {/* Below the name, as it was: the run's own file is not something to go looking for. */}
      <div className="flex gap-2 pt-0.5">{action}</div>
      {open && children}
    </div>
  )
}

/** The finished LoRA. Always on the card, never behind the collapse: it is what the run produced. */
function DownloadLora({ run }: { run: TrainingRun }): React.JSX.Element {
  return (
    <a
      href={`${window.location.origin}/download/lora/${run.id}`}
      download={loraName(run)}
      title={`Download ${loraName(run)}`}
      className="flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-panel"
    >
      <DownloadIcon className="h-3 w-3" /> Download .safetensors
    </a>
  )
}

function ResumeButton({ run }: { run: TrainingRun }): React.JSX.Element {
  const resume = useTrainingStore((s) => s.resume)
  const runs = useTrainingStore((s) => s.runs)
  const blocked = runs.some((r) => r.status === 'training' || r.status === 'queued')
  return (
    <button
      onClick={() => void resume(run.id)}
      disabled={blocked}
      title={blocked ? 'Another run is using the GPU' : 'Resume from the last checkpoint'}
      className="flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-panel disabled:opacity-40"
    >
      <PlayIcon className="h-3 w-3" /> Resume
    </button>
  )
}

/**
 * The mid-run LoRAs a run wrote, so a run can be judged (or salvaged) without waiting for it to
 * finish. They live in the project's working dir, which no model picker scans, so "Add to models"
 * copies one into `models/loras/` where a Load LoRA node can select it.
 */
function SnapshotRows({ run }: { run: TrainingRun }): React.JSX.Element | null {
  const snapshots = useTrainingStore((s) => s.snapshotsByRun[run.id])
  const exportSnapshot = useTrainingStore((s) => s.exportSnapshot)
  const [added, setAdded] = useState<Record<number, string>>({})

  if (!snapshots || snapshots.length === 0) return null
  return (
    <div className="flex flex-col">
      <div className="pb-0.5 pl-4 pt-1 text-[10px] uppercase tracking-wide text-zinc-600">
        Snapshots
      </div>
      {snapshots.map((snap) => (
        <div key={snap.step} className="flex items-center gap-2 py-1 pl-4">
          <div className="min-w-0 flex-1">
            <span className="block truncate text-[11px] text-zinc-200" title={snapshotName(snap)}>
              {snapshotName(snap)}
            </span>
            <span className="text-[10px] text-zinc-500">
              {formatSize(snap.sizeBytes)}
              {(added[snap.step] ?? snap.loraPath)
                ? ` · added as ${(added[snap.step] ?? snap.loraPath ?? '').split('/').pop()}`
                : ''}
            </span>
          </div>
          <button
            onClick={() =>
              void exportSnapshot(run.id, snap.step).then((path) => {
                if (path) setAdded((prev) => ({ ...prev, [snap.step]: path }))
              })
            }
            disabled={!!(added[snap.step] ?? snap.loraPath)}
            title="Copy into models/loras so a Load LoRA node can pick it"
            className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-panel disabled:opacity-40"
          >
            {(added[snap.step] ?? snap.loraPath) ? 'Added' : 'Add to models'}
          </button>
          <a
            href={`${window.location.origin}/download/snapshot/${run.id}/${snap.step}`}
            download={snapshotName(snap)}
            title="Download this snapshot"
            className="shrink-0 rounded border border-border p-1 text-emerald-300 hover:bg-panel"
          >
            <DownloadIcon className="h-3 w-3" />
          </a>
        </div>
      ))}
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
            <RunCard
              key={r.id}
              run={r}
              title={loraName(r)}
              subtitle={`rank ${r.hyperparams.rank} · ${r.totalSteps} steps · ${r.hyperparams.resolution}px · ${when(r.updatedAt)}`}
              action={<DownloadLora run={r} />}
            >
              <SnapshotRows run={r} />
            </RunCard>
          ))}
        </>
      )}
      {resumable.length > 0 && (
        <>
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
            Resumable ({resumable.length})
          </div>
          {resumable.map((r) => (
            <RunCard
              key={r.id}
              run={r}
              title={r.name}
              subtitle={`stopped at ${r.step}/${r.totalSteps} · ${when(r.updatedAt)}`}
              action={<ResumeButton run={r} />}
              note={r.error}
            >
              <SnapshotRows run={r} />
            </RunCard>
          ))}
        </>
      )}
    </div>
  )
}
