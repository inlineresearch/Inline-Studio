/**
 * The consent gate. Findings grouped by severity in plain language, with a typed confirmation for
 * anything needing approval.
 *
 * The copy is deliberate: an extension runs in the same process as Inline Studio, so a clean scan
 * is not a safety guarantee and must never be presented as one.
 */
import { useState } from 'react'
import type { ScanReport, ScanFinding, FindingSeverity } from '@shared/extensions'

const TONE: Record<FindingSeverity, string> = {
  critical: 'border-red-500/40 bg-red-500/10 text-red-300',
  high: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  medium: 'border-amber-500/30 bg-amber-500/5 text-amber-200/90',
  low: 'border-border bg-surface/60 text-zinc-400',
}

const HEADING: Record<FindingSeverity, string> = {
  critical: 'Blocked',
  high: 'Needs your approval',
  medium: 'Needs your approval',
  low: 'For your information',
}

function FindingRow({ finding }: { finding: ScanFinding }): React.JSX.Element {
  return (
    <div className={`rounded-md border px-3 py-2 ${TONE[finding.severity]}`}>
      <div className="text-[12px] leading-relaxed">{finding.message}</div>
      {finding.file && (
        <div className="mt-1 font-mono text-[10px] opacity-70">
          {finding.file}
          {finding.line > 0 && `:${finding.line}`}
        </div>
      )}
    </div>
  )
}

export function SecurityReport({
  name,
  report,
  onApprove,
  onCancel,
}: {
  name: string
  report: ScanReport
  onApprove: (rules: string[]) => void
  onCancel: () => void
}): React.JSX.Element {
  const [typed, setTyped] = useState('')
  const groups: FindingSeverity[] = ['critical', 'high', 'medium', 'low']
  const confirmed = typed.trim() === name

  return (
    <div className="flex flex-col gap-4 p-5">
      <div>
        <h3 className="text-sm font-semibold text-zinc-100">
          {report.blocked ? `${name} cannot be installed` : `Review ${name} before installing`}
        </h3>
        <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
          Extensions run with the same access as Inline Studio itself. They can read and write your
          files and reach the network. This review flags what the code does; it cannot prevent it.
          Only install extensions from authors you trust.
        </p>
      </div>

      {groups.map((severity) => {
        const items = report.findings.filter((f) => f.severity === severity)
        if (items.length === 0) return null
        return (
          <section key={severity} className="flex flex-col gap-1.5">
            <h4 className="text-[11px] font-semibold uppercase tracking-wide text-zinc-400">
              {HEADING[severity]}
            </h4>
            {items.map((finding, i) => (
              <FindingRow key={`${finding.rule}-${finding.file}-${i}`} finding={finding} />
            ))}
          </section>
        )
      })}

      {report.blocked ? (
        <div className="flex justify-end">
          <button
            onClick={onCancel}
            className="rounded-md border border-border px-3 py-1.5 text-[11px] font-medium text-zinc-300 hover:bg-surface hover:text-zinc-100"
          >
            Close
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-2 border-t border-border pt-4">
          <label className="text-[11px] text-zinc-400">
            Type <span className="font-mono text-zinc-200">{name}</span> to confirm you have read
            this.
          </label>
          <div className="flex items-center gap-2">
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              placeholder={name}
              spellCheck={false}
              className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2.5 py-1.5 font-mono text-[12px] text-zinc-100 outline-none focus:border-zinc-500"
            />
            <button
              onClick={onCancel}
              className="rounded-md border border-border px-3 py-1.5 text-[11px] font-medium text-zinc-300 hover:bg-surface hover:text-zinc-100"
            >
              Cancel
            </button>
            <button
              onClick={() => onApprove(report.consentRules)}
              disabled={!confirmed}
              className="rounded-md bg-emerald-500/90 px-3 py-1.5 text-[11px] font-semibold text-black hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Install anyway
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
