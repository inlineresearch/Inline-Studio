/**
 * The run status control, pinned in the header just before the settings gear.
 *
 * It reads the whole process, not the open project: Core keeps working through a project switch or
 * a tab switch, so the count here can include runs from a project that is not currently open, and
 * runs submitted straight to the Core API rather than started from this UI.
 */
import { useEffect, useRef, useState } from 'react'
import type { ActivityRun } from '@shared/types'
import { useActivityStore } from '../../store/activityStore'
import { ActivityIcon, CloseIcon, StopCircleIcon } from '../../components/icons'

function statusTone(status: ActivityRun['status']): string {
  if (status === 'done') return 'text-emerald-400'
  if (status === 'error') return 'text-red-400'
  if (status === 'running') return 'text-emerald-400'
  if (status === 'queued') return 'text-amber-400'
  return 'text-zinc-500'
}

function statusLabel(run: ActivityRun): string {
  if (run.status === 'queued') {
    return run.queuePosition === null ? 'Queued' : `Queued, ${run.queuePosition + 1} in line`
  }
  if (run.status === 'running') {
    if (run.statusLabel) return run.statusLabel
    return run.fraction === null ? 'Working' : `${Math.round(run.fraction * 100)}%`
  }
  return run.status.charAt(0).toUpperCase() + run.status.slice(1)
}

function RunRow({
  run,
  onCancel,
}: {
  run: ActivityRun
  onCancel?: (runId: string) => void
}): React.JSX.Element {
  return (
    <li className="flex items-center gap-2 px-3 py-1.5 hover:bg-black/20">
      <span className={`text-[10px] leading-none ${statusTone(run.status)}`}>&#9679;</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-1.5">
          <span className="truncate text-[11px] text-zinc-200">{run.title}</span>
          {run.origin === 'api' && (
            <span className="shrink-0 rounded bg-zinc-700 px-1 text-[9px] text-zinc-300">API</span>
          )}
        </div>
        <div className="truncate text-[10px] text-zinc-500">
          {statusLabel(run)}
          {run.projectName ? ` · ${run.projectName}` : ''}
        </div>
        {run.status === 'running' && run.fraction !== null && (
          <div className="mt-1 h-px w-full bg-zinc-700">
            <div
              className="h-px bg-emerald-500"
              style={{ width: `${Math.round(run.fraction * 100)}%` }}
            />
          </div>
        )}
      </div>
      {onCancel && (
        <button
          onClick={() => onCancel(run.runId)}
          title="Cancel this run"
          aria-label="Cancel this run"
          className="shrink-0 rounded p-1 text-zinc-500 hover:bg-black/40 hover:text-red-400"
        >
          <StopCircleIcon className="h-3.5 w-3.5" />
        </button>
      )}
    </li>
  )
}

function Section({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="border-b border-border last:border-b-0">
      <div className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
        {title}
      </div>
      <ul>{children}</ul>
    </div>
  )
}

export function ActivityIndicator(): React.JSX.Element {
  const live = useActivityStore((s) => s.live)
  const history = useActivityStore((s) => s.history)
  const cancel = useActivityStore((s) => s.cancel)
  const cancelAll = useActivityStore((s) => s.cancelAll)
  const clearHistory = useActivityStore((s) => s.clearHistory)
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  const running = live.filter((r) => r.status === 'running')
  const queued = live.filter((r) => r.status === 'queued')
  const busy = running.length > 0

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent): void => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const summary = busy
    ? `${running.length} running${queued.length ? `, ${queued.length} queued` : ''}`
    : 'Nothing running'

  return (
    <div ref={rootRef} className="group relative">
      <div
        className={`flex h-8 items-center rounded-md border transition-colors ${
          open
            ? 'border-zinc-600 bg-panel'
            : 'border-border bg-surface hover:border-zinc-600 hover:bg-panel'
        }`}
      >
        <button
          onClick={() => setOpen(!open)}
          title={summary}
          aria-label={`Active runs: ${live.length}`}
          aria-expanded={open}
          className="flex h-full items-center gap-1.5 rounded-l-md pl-2 pr-1.5"
        >
          <span className="relative flex h-4 w-4 items-center justify-center">
            <ActivityIcon className={`h-4 w-4 ${busy ? 'text-emerald-400' : 'text-zinc-500'}`} />
            {busy && (
              <span className="absolute inset-0 animate-ping rounded-full bg-emerald-500/20" />
            )}
          </span>
          <span className="text-[11px] font-medium text-zinc-300">Active Runs</span>
          <span
            className={`min-w-[1.25rem] rounded px-1 text-center text-[11px] font-semibold tabular-nums ${
              busy ? 'bg-emerald-500/15 text-emerald-300' : 'bg-black/30 text-zinc-400'
            }`}
          >
            {live.length}
          </span>
        </button>
        {/* Revealed on hover so the destructive action is never the thing under an idle cursor. */}
        <button
          onClick={() => void cancelAll()}
          disabled={live.length === 0}
          title="Cancel all runs"
          aria-label="Cancel all runs"
          className="mr-1 flex h-6 w-6 items-center justify-center rounded text-red-400 opacity-0 transition-opacity hover:bg-red-500/15 disabled:cursor-not-allowed disabled:text-zinc-600 group-hover:opacity-100 focus-visible:opacity-100"
        >
          <CloseIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 max-h-[28rem] w-80 overflow-y-auto rounded-md border border-border bg-panel shadow-2xl">
          {running.length > 0 && (
            <Section title="Running">
              {running.map((r) => (
                <RunRow key={r.runId} run={r} onCancel={cancel} />
              ))}
            </Section>
          )}
          {queued.length > 0 && (
            <Section title="Queued">
              {queued.map((r) => (
                <RunRow key={r.runId} run={r} onCancel={cancel} />
              ))}
            </Section>
          )}
          {history.length > 0 && (
            <Section title="Recent">
              {history.map((r) => (
                <RunRow key={r.runId} run={r} />
              ))}
            </Section>
          )}
          {live.length === 0 && history.length === 0 && (
            <p className="px-3 py-4 text-center text-[11px] text-zinc-500">
              Nothing has run in this project yet.
            </p>
          )}
          {history.length > 0 && (
            <div className="flex justify-end border-t border-border px-2 py-1.5">
              <button
                onClick={() => void clearHistory()}
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-black/30 hover:text-zinc-300"
              >
                <CloseIcon className="h-3 w-3" />
                Clear history
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
