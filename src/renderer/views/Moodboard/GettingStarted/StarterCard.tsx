import type { StarterRecipe } from '../../../lib/starterRecipes'
import type { Advice, Tier } from '../../../lib/vramAdvice'

/** Tier colours. Amber is as loud as this gets, because every tier still runs. */
const TIER_STYLE: Record<Tier, { chip: string; label: string }> = {
  best: { chip: 'bg-emerald-500/15 text-emerald-200', label: 'Runs great' },
  good: { chip: 'bg-emerald-500/10 text-emerald-300', label: 'Runs well' },
  ok: { chip: 'bg-sky-500/15 text-sky-200', label: 'Runs' },
  heavy: { chip: 'bg-amber-500/15 text-amber-200', label: 'Heavy' },
}

/**
 * One hue per kind of generation, so the list is scannable before it is read. Deliberately clear of
 * the emerald/sky/amber the tier chips already own, and of the yellow-green accent.
 */
const KIND_STYLE: Record<StarterRecipe['kind'], { tile: string; label: string; text: string }> = {
  image: { tile: 'bg-indigo-500/15 text-indigo-300', label: 'Image', text: 'text-indigo-300' },
  video: { tile: 'bg-fuchsia-500/15 text-fuchsia-300', label: 'Video', text: 'text-fuchsia-300' },
  training: { tile: 'bg-teal-500/15 text-teal-300', label: 'Training', text: 'text-teal-300' },
}

export interface CardStatus {
  /** Model files still to download. 0 once everything is installed. */
  missing: number
  /** An extra line from the engine's own fit estimate, only trustworthy once installed. */
  note: string | null
}

const CHIP = 'shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium leading-none'

/** One getting-started card: a compact row, presentational, reports clicks upward. */
export function StarterCard({
  icon,
  title,
  tag,
  kind,
  blurb,
  advice,
  recommended,
  recommendedLabel,
  status,
  action,
  onPick,
  onGetModels,
  disabled = false,
}: {
  icon: React.JSX.Element
  title: string
  /** Short chip beside the title, e.g. `New`. */
  tag?: string
  kind: StarterRecipe['kind']
  blurb: string
  advice: Advice
  recommended: boolean
  recommendedLabel: string
  /** Null while the install state is still unknown, so nothing flashes on first paint. */
  status: CardStatus | null
  /** A one-off setup step this card needs, shown in the same slot as the model downloads. */
  action?: { hint: string; label: string; onClick: () => void }
  onPick: () => void
  onGetModels?: () => void
  disabled?: boolean
}): React.JSX.Element {
  const tier = advice.tier ? TIER_STYLE[advice.tier] : null
  const kindStyle = KIND_STYLE[kind]
  return (
    <div
      className={`group flex items-center gap-3 rounded-lg border bg-surface/95 px-3 py-2.5 text-left shadow-sm backdrop-blur transition ${
        recommended ? 'border-accent/50' : 'border-border'
      } ${disabled ? 'opacity-50' : 'hover:border-zinc-500 hover:bg-surface'}`}
    >
      <button
        onClick={onPick}
        disabled={disabled}
        className="flex min-w-0 flex-1 items-center gap-3 text-left disabled:cursor-not-allowed"
      >
        <span
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${kindStyle.tile}`}
        >
          {icon}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-1.5">
            <span className="text-sm font-semibold leading-tight text-zinc-50">{title}</span>
            {/* Solid accent for `New`, outlined for the recommendation: on a card that is both,
                one has to win, and `New` is the thing worth noticing. */}
            {tag && <span className={`${CHIP} bg-accent font-semibold text-black`}>{tag}</span>}
            {tier && <span className={`${CHIP} ${tier.chip}`}>{tier.label}</span>}
            {/* Inline, not a floating ribbon: the cards stack tightly and an overhanging badge
                would sit on top of the card above it. */}
            {recommended && (
              <span className={`${CHIP} border border-accent/40 bg-accent/10 text-accent`}>
                {recommendedLabel}
              </span>
            )}
          </span>
          <span className="mt-1 block text-xs leading-snug text-zinc-300">{blurb}</span>
          <span className="mt-0.5 block text-[11px] leading-snug text-zinc-400">
            <span className={`font-semibold uppercase tracking-wide ${kindStyle.text}`}>
              {kindStyle.label}
            </span>
            <span className="text-zinc-600"> · </span>
            {advice.note}
          </span>
          {status?.note && (
            <span className="mt-0.5 block text-[11px] leading-snug text-amber-200/80">
              {status.note}
            </span>
          )}
        </span>
      </button>
      {status && status.missing > 0 && (
        <FooterAction
          hint={`${status.missing} ${status.missing === 1 ? 'file' : 'files'}`}
          label="Get models"
          onClick={onGetModels}
        />
      )}
      {action && <FooterAction hint={action.hint} label={action.label} onClick={action.onClick} />}
    </div>
  )
}

/** The right-hand setup slot: what is missing, and the one button that fixes it. */
function FooterAction({
  hint,
  label,
  onClick,
}: {
  hint: string
  label: string
  onClick?: () => void
}): React.JSX.Element {
  return (
    <div className="flex shrink-0 items-center gap-2 self-stretch border-l border-border pl-3">
      <span className="text-[11px] text-zinc-400">{hint}</span>
      {onClick && (
        <button
          onClick={onClick}
          className="rounded border border-border px-2 py-1 text-[11px] font-medium text-zinc-200 transition hover:border-zinc-500 hover:bg-panel"
        >
          {label}
        </button>
      )}
    </div>
  )
}
