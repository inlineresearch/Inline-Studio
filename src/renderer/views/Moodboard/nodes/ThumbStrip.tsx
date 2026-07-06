import { AudioGlyph } from './NodeBadge'

/** One selectable media thumbnail in the strip. */
export interface StripItem {
  id: string
  /** The media URL (a playable video / image src). */
  url: string
  kind: 'image' | 'video' | 'audio'
  /** Poster still for a video, so we can render a cheap <img> instead of a heavy <video>. */
  poster?: string
}

/**
 * A horizontal strip of small rounded thumbnails, shown only when a node holds more than one
 * media item. Clicking one selects it as the node's main (large) preview. Floats over the bottom
 * of the media on a soft scrim, and scrolls horizontally when there are many.
 */
export function ThumbStrip({
  items,
  selected,
  onSelect,
}: {
  items: StripItem[]
  selected: number
  onSelect: (index: number) => void
}): React.JSX.Element | null {
  if (items.length <= 1) return null
  return (
    <div className="nodrag nowheel absolute inset-x-0 bottom-0 z-10 flex gap-1 overflow-x-auto bg-gradient-to-t from-black/80 via-black/40 to-transparent px-1.5 pb-1.5 pt-5">
      {items.map((it, i) => (
        <button
          key={it.id}
          onClick={(e) => {
            e.stopPropagation()
            onSelect(i)
          }}
          title={`View ${i + 1} of ${items.length}`}
          className={`h-8 w-8 shrink-0 overflow-hidden rounded-md border-2 bg-black/60 transition ${
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
            <video src={it.url} muted preload="metadata" className="h-full w-full object-cover" />
          ) : (
            <img src={it.poster ?? it.url} alt="" className="h-full w-full object-cover" />
          )}
        </button>
      ))}
    </div>
  )
}
