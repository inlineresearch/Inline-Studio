import type { Advice, Tier } from '../../../lib/vramAdvice'

/** Tier colours. Amber is as loud as this gets, because every tier still runs. */
const TIER_STYLE: Record<Tier, { chip: string; label: string }> = {
  best: { chip: 'bg-emerald-500/15 text-emerald-300', label: 'Runs great' },
  good: { chip: 'bg-emerald-500/10 text-emerald-400/90', label: 'Runs well' },
  ok: { chip: 'bg-sky-500/10 text-sky-300', label: 'Runs' },
  heavy: { chip: 'bg-amber-500/10 text-amber-300', label: 'Heavy' },
}

export interface CardStatus {
  /** Model files still to download. 0 once everything is installed. */
  missing: number
  /** An extra line from the engine's own fit estimate, only trustworthy once installed. */
  note: string | null
}

/** One getting-started card: presentational, reports clicks upward. */
export function StarterCard({
  icon,
  title,
  blurb,
  advice,
  recommended,
  recommendedLabel,
  status,
  onPick,
  onGetModels,
  disabled = false,
}: {
  icon: React.JSX.Element
  title: string
  blurb: string
  advice: Advice
  recommended: boolean
  recommendedLabel: string
  /** Null while the install state is still unknown, so nothing flashes on first paint. */
  status: CardStatus | null
  onPick: () => void
  onGetModels?: () => void
  disabled?: boolean
}): React.JSX.Element {
  const tier = advice.tier ? TIER_STYLE[advice.tier] : null
  return (
    <div
      className={`relative flex flex-col rounded-lg border bg-surface/80 p-3 text-left transition ${
        recommended ? 'border-accent/60' : 'border-border'
      } ${disabled ? 'opacity-50' : 'hover:border-zinc-500'}`}
    >
      {recommended && (
        <span className="absolute -top-2 left-3 rounded-full bg-accent px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-black">
          {recommendedLabel}
        </span>
      )}
      <button
        onClick={onPick}
        disabled={disabled}
        className="flex flex-col gap-1.5 text-left disabled:cursor-not-allowed"
      >
        <span className="flex items-center gap-2">
          <span className="text-zinc-400">{icon}</span>
          <span className="text-sm font-medium text-zinc-100">{title}</span>
          {tier && (
            <span className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${tier.chip}`}>
              {tier.label}
            </span>
          )}
        </span>
        <span className="text-[11px] leading-relaxed text-zinc-400">{blurb}</span>
        <span className="text-[10px] leading-relaxed text-zinc-500">{advice.note}</span>
      </button>
      {status && status.missing > 0 && (
        <div className="mt-2 flex items-center gap-2 border-t border-border pt-2">
          <span className="flex-1 text-[10px] text-zinc-500">
            {status.missing} model {status.missing === 1 ? 'file' : 'files'} to download
          </span>
          {onGetModels && (
            <button
              onClick={onGetModels}
              className="rounded px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-panel hover:text-zinc-200"
            >
              Get models
            </button>
          )}
        </div>
      )}
      {status?.note && (
        <p className="mt-2 border-t border-border pt-2 text-[10px] text-zinc-500">{status.note}</p>
      )}
    </div>
  )
}
