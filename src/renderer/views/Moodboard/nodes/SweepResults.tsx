/** A reference sweep's progress and verdict, on the Finetune node's face. */
import { useEffect, useState } from 'react'
import type { SweepResult } from '@shared/types'
import { useTuningStore } from '../../../store/tuningStore'
import { resolveMedia } from '@/lib/media'

/** Coarse on purpose: the thresholds behind these are uncalibrated. */
function verdictClass(verdict: string): string {
  if (verdict === 'keep') return 'text-emerald-300'
  if (verdict === 'consider-removing') return 'text-red-300'
  // Costs identity, pays for it in clothes: a different decision from one that pays nothing.
  if (verdict === 'keeps-the-wardrobe') return 'text-amber-300'
  return 'text-zinc-400'
}

/** Progress as counts, never a bar: a sweep's cells are not evenly expensive. */
function ProgressLine({
  done,
  total,
  result,
}: {
  done: number
  total: number
  result?: SweepResult
}): React.JSX.Element {
  const best = result?.combinations?.[0]
  return (
    <span className="text-[10px] text-zinc-400">
      {done}/{total} renders
      {result ? ` · ${result.refs} refs` : ''}
      {best ? ` · best ${best.key} ${best.adjusted.toFixed(0)} (±${best.spread.toFixed(0)})` : ''}
    </span>
  )
}

export function SweepResults({ runId }: { runId: string | null }): React.JSX.Element | null {
  const progress = useTuningStore((s) => (runId ? s.progressByRun[runId] : undefined))
  const result = useTuningStore((s) => (runId ? s.resultByRun[runId] : undefined))
  const error = useTuningStore((s) => (runId ? s.errorByRun[runId] : undefined))
  const load = useTuningStore((s) => s.load)
  const [open, setOpen] = useState(false)

  // A finished sweep's events are long gone by the time a reloaded page mounts this.
  useEffect(() => {
    if (runId) void load(runId)
  }, [runId, load])

  if (!runId) return null
  if (error) return <div className="px-2 pb-1.5 text-[10px] text-red-300">{error}</div>
  if (!progress && !result) {
    return <div className="px-2 pb-1.5 text-[10px] text-zinc-500">Waiting for the sweep…</div>
  }

  const done = result?.done ?? progress?.done ?? 0
  const total = result?.total ?? progress?.total ?? 0
  const running = !result || result.status === 'running' || result.status === 'queued'

  return (
    <div className="nodrag border-t border-border px-2 py-1.5">
      <div className="flex items-center justify-between gap-2">
        <ProgressLine done={done} total={total} result={result} />
        {result?.report && !running && (
          // Served from the project folder, so the report opens in a tab rather than being a path
          // the user has to go and find on disk.
          <a
            href={resolveMedia(result.report)}
            target="_blank"
            rel="noreferrer"
            title={result.report}
            className="shrink-0 rounded border border-border px-1.5 text-[10px] text-emerald-300 hover:border-emerald-500 hover:text-emerald-200"
          >
            Report
          </a>
        )}
        {result && !running && (
          <button
            onClick={() => setOpen((v) => !v)}
            className="shrink-0 rounded border border-border px-1.5 text-[10px] text-zinc-300 hover:border-zinc-500 hover:text-white"
          >
            {open ? 'Hide' : 'Details'}
          </button>
        )}
      </div>

      {result && !running && (
        <div className="mt-1 text-[11px] text-zinc-200">{result.headline}</div>
      )}
      {/* Never dropped, and never smaller than the number above it. */}
      {result && <div className="mt-0.5 text-[9px] text-amber-300/80">{result.caveat}</div>}

      {open && result && (
        <div className="nowheel mt-2 max-h-64 overflow-auto">
          <div className="mb-1 text-[9px] uppercase tracking-wide text-zinc-500">
            What each reference was worth
          </div>
          <table className="w-full text-[10px]">
            <thead className="text-zinc-500">
              <tr>
                <th className="text-left font-normal">Ref</th>
                <th className="text-right font-normal">Delta</th>
                <th className="text-right font-normal">±</th>
                <th className="text-right font-normal">Framing</th>
                <th className="text-right font-normal">Cloth</th>
                <th className="text-left font-normal">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {result.contributions.map((c) => (
                <tr key={c.index} className="border-t border-border/50">
                  <td>{c.index}</td>
                  <td className="text-right tabular-nums">
                    {c.delta > 0 ? '+' : ''}
                    {c.delta.toFixed(1)}
                  </td>
                  <td className="text-right tabular-nums text-zinc-500">{c.spread.toFixed(1)}</td>
                  <td className="text-right tabular-nums text-zinc-500">
                    {c.framingShift > 0 ? '+' : ''}
                    {c.framingShift.toFixed(2)}pp
                  </td>
                  {/* A dash, never a zero: the clothes may simply never have been in frame. */}
                  <td className="text-right tabular-nums text-zinc-500">
                    {c.wardrobeDelta === null
                      ? '-'
                      : `${c.wardrobeDelta > 0 ? '+' : ''}${c.wardrobeDelta.toFixed(1)}`}
                  </td>
                  <td className={verdictClass(c.verdict)}>{c.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-1 text-[9px] text-zinc-500">
            Suggestions only. Edit the character through Encode or Verify, never from here.
          </div>

          {(result.confirmations?.length ?? 0) > 0 && (
            <>
              <div className="mb-1 mt-3 text-[9px] uppercase tracking-wide text-zinc-500">
                Re-tested on fresh seeds
              </div>
              <table className="w-full text-[10px]">
                <thead className="text-zinc-500">
                  <tr>
                    <th className="text-left font-normal">Ref</th>
                    <th className="text-right font-normal">Sweep</th>
                    <th className="text-right font-normal">Re-test</th>
                    <th className="text-right font-normal">Pooled</th>
                    <th className="text-left font-normal">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.confirmations ?? []).map((c) => (
                    <tr key={c.index} className="border-t border-border/50">
                      <td>{c.index}</td>
                      <td className="text-right tabular-nums text-zinc-500">
                        {c.initial > 0 ? '+' : ''}
                        {c.initial.toFixed(1)}
                      </td>
                      <td className="text-right tabular-nums text-zinc-500">
                        {c.retest > 0 ? '+' : ''}
                        {c.retest.toFixed(1)}
                      </td>
                      {/* The one to act on: every pair from both passes together. */}
                      <td className="text-right tabular-nums">
                        {c.pooled > 0 ? '+' : ''}
                        {c.pooled.toFixed(1)}
                      </td>
                      <td
                        className={
                          c.verdict === 'removal confirmed' ? 'text-red-300' : 'text-zinc-400'
                        }
                      >
                        {c.verdict}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          <div className="mb-1 mt-3 text-[9px] uppercase tracking-wide text-zinc-500">
            Per prompt · read down a column, not across
          </div>
          {result.contributions.map((c) => (
            <div key={c.index} className="flex gap-2 text-[10px]">
              <span className="w-8 shrink-0 text-zinc-500">ref {c.index}</span>
              <span className="min-w-0 flex-1 truncate">
                {Object.entries(c.perPrompt)
                  .map(([p, v]) => `${p.slice(0, 18)} ${v > 0 ? '+' : ''}${v.toFixed(1)}`)
                  .join(' · ')}
              </span>
            </div>
          ))}

          {result.report && (
            <div className="mt-3 break-all text-[9px] text-zinc-500">report: {result.report}</div>
          )}
          {result.spent > 0 && (
            <div className="text-[9px] text-zinc-500">
              estimated spend ${result.spent.toFixed(2)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
