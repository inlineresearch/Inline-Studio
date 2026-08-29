import { AudioGlyph, XIcon } from './NodeBadge'
import { scoreSuffix, scoreTitle, scoreTone } from '@/lib/continuity'

/** One selectable media thumbnail in the strip. */
export interface StripItem {
  id: string
  /** The media URL (a playable video / image src). */
  url: string
  kind: 'image' | 'video' | 'audio'
  /** Poster still for a video, so we can render a cheap <img> instead of a heavy <video>. */
  poster?: string
  /** Which input port this feeds, on models that have more than one of a kind. */
  label?: string
  /** 0-100 identity match, when a character was applied. Absent means "not measured", which is a
   *  different fact from a low score, so it must never be shown as zero. */
  score?: number
  /** The score covers the face alone, because the references cannot judge this take's framing. */
  faceOnly?: boolean
}

/**
 * A horizontal strip of small rounded thumbnails floating over the media. Two modes:
 *  - Selection (default): shown only when there's more than one item; clicking one selects it as
 *    the node's main (large) preview. Used for a node's take history.
 *  - Manage (`onRemove` given): shown for one or more items, each with a hover × that removes it.
 *    Used for a node's inputs. Pass `edge="top"` to dock it at the top (so it doesn't collide with
 *    a bottom take strip). Scrolls horizontally when there are many.
 */
export function ThumbStrip({
  items,
  selected,
  onSelect,
  onRemove,
  edge = 'bottom',
  leading,
}: {
  items: StripItem[]
  selected?: number
  onSelect?: (index: number) => void
  /** When provided, each thumbnail gets a hover × that calls this with its index. */
  onRemove?: (index: number) => void
  edge?: 'top' | 'bottom'
  /** A tile pinned before the items, owning no media of its own. Used for the Current slot, whose
   *  presence is also what forces the strip open on a node with fewer than two takes. */
  leading?: React.ReactNode
}): React.JSX.Element | null {
  const manage = !!onRemove
  if (items.length === 0 && !leading) return null
  if (!manage && !leading && items.length <= 1) return null
  const scrim =
    edge === 'top'
      ? 'top-0 bg-gradient-to-b from-black/80 via-black/40 to-transparent pt-1.5 pb-5'
      : 'bottom-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent pb-1.5 pt-5'
  // Wraps rather than scrolling sideways: a single row fits about six thumbnails at the default node
  // width, which hid the rest of a large input set behind a scrollbar nobody found. Rows grow as the
  // node is resized, capped so the strip never swallows the preview it floats over.
  return (
    <div
      className={`nodrag nowheel absolute inset-x-0 z-10 flex max-h-[55%] flex-wrap gap-1 overflow-y-auto px-1.5 ${scrim}`}
    >
      {leading}
      {items.map((it, i) => (
        <div key={it.id} className="group/thumb relative shrink-0">
          <button
            onClick={(e) => {
              e.stopPropagation()
              onSelect?.(i)
            }}
            title={
              manage
                ? `${it.label ?? 'Input'} ${i + 1} of ${items.length}`
                : `View ${i + 1} of ${items.length}`
            }
            className={`h-8 w-8 overflow-hidden rounded-md border-2 bg-black/60 transition ${
              i === selected
                ? 'border-accent opacity-100'
                : 'border-white/25 opacity-70 hover:opacity-100'
            }`}
          >
            {it.kind === 'audio' ? (
              <span className="flex h-full w-full items-center justify-center text-emerald-400">
                <AudioGlyph className="h-4 w-4" />
              </span>
            ) : it.kind === 'video' && !it.poster ? (
              // The `#t=` fragment seeks the browser to one frame and paints it. Without it,
              // preload="metadata" decodes nothing and an un-postered clip renders as a black tile.
              <video
                src={`${it.url}#t=0.1`}
                muted
                preload="metadata"
                className="h-full w-full object-cover"
              />
            ) : (
              <img src={it.poster ?? it.url} alt="" className="h-full w-full object-cover" />
            )}
          </button>
          {it.label && (
            <span className="pointer-events-none absolute inset-x-0 bottom-0 truncate rounded-b-md bg-black/80 px-0.5 text-center text-[7px] leading-tight text-zinc-300">
              {it.label}
            </span>
          )}
          {it.score !== undefined && (
            <span
              title={scoreTitle(it.score, it.faceOnly)}
              className={`pointer-events-none absolute left-0.5 top-0.5 rounded bg-black/85 px-1 text-[7px] leading-tight ${scoreTone(it.score)}`}
            >
              {Math.round(it.score)}
              {scoreSuffix(it.faceOnly)}
            </span>
          )}
          {onRemove && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onRemove(i)
              }}
              title="Remove input"
              aria-label="Remove input"
              className="nodrag absolute right-0.5 top-0.5 hidden h-4 w-4 items-center justify-center rounded-full border border-border bg-black/85 text-zinc-200 shadow hover:bg-black hover:text-white group-hover/thumb:flex"
            >
              <XIcon className="h-2.5 w-2.5" />
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
