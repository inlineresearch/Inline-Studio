import { useRef, useState } from 'react'
import { studio } from '@/lib/studio'
import type { WorkflowSummary } from '@shared/types'
import { DownloadIcon, ExternalLinkIcon, EyeIcon, ImportIcon } from '../../components/icons'

const COUNT = 'flex items-center gap-1 text-[10px] text-zinc-500'
// Absolute, not `h-full`: an intrinsically tall image resolves `h-full` against itself and
// stretches the aspect box, so image cards came out taller than video ones.
const MEDIA = 'absolute inset-0 h-full w-full object-cover'

function compact(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)
}

/** One catalogue card: hero, title, counts, Import, and a link out to the write-up. */
export function WorkflowCard({
  card,
  importing,
  onImport,
}: {
  card: WorkflowSummary
  importing: boolean
  onImport: () => void
}): React.JSX.Element {
  const video = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)

  // Hover-play rather than autoplay: a grid of looping video over a remote origin is heavy on a
  // machine that is also mid-render.
  const onEnter = (): void => {
    if (card.heroType !== 'video') return
    setPlaying(true)
    void video.current?.play().catch(() => setPlaying(false))
  }
  const onLeave = (): void => {
    setPlaying(false)
    video.current?.pause()
  }

  return (
    <div
      className="flex flex-col overflow-hidden rounded-lg border border-border bg-panel"
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      <div className="relative aspect-video w-full shrink-0 overflow-hidden bg-black">
        {card.heroUrl ? (
          card.heroType === 'video' ? (
            <video
              ref={video}
              // Without a poster the tile would be black until hover, so the fragment seeks the
              // browser to the first frame and metadata is enough to paint it.
              src={card.heroPosterUrl ? card.heroUrl : `${card.heroUrl}#t=0.1`}
              poster={card.heroPosterUrl ?? undefined}
              muted
              loop
              playsInline
              preload={card.heroPosterUrl ? 'none' : 'metadata'}
              className={MEDIA}
            />
          ) : (
            <img src={card.heroUrl} alt="" className={MEDIA} />
          )
        ) : null}

        {card.heroType === 'video' && !playing ? (
          <span className="pointer-events-none absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1 py-0.5 text-[9px] leading-none text-zinc-300">
            Video
          </span>
        ) : null}

        <button
          onClick={onImport}
          disabled={importing}
          className="absolute right-1.5 top-1.5 rounded border border-emerald-700 bg-emerald-600/80 px-2 py-0.5 text-[11px] font-medium text-white transition-colors hover:bg-emerald-600 disabled:opacity-60"
        >
          {importing ? 'Importing…' : 'Import'}
        </button>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-1 p-2">
        <h3 className="truncate text-[12px] font-semibold text-zinc-100" title={card.title}>
          {card.title}
        </h3>
        <p className="line-clamp-2 text-[10px] leading-snug text-zinc-500">{card.summary}</p>

        <div className="mt-auto flex items-center gap-2 pt-1.5">
          <span className={COUNT} title="Views">
            <EyeIcon className="h-3 w-3" />
            {compact(card.viewCount)}
          </span>
          <span className={COUNT} title="Downloaded from inlinestudio.art">
            <DownloadIcon className="h-3 w-3" />
            {compact(card.downloadCount)}
          </span>
          <span className={COUNT} title="Imported into Inline Studio">
            <ImportIcon className="h-3 w-3" />
            {compact(card.importCount)}
          </span>
          {card.pageUrl ? (
            <button
              onClick={() => void studio().shell.openExternal(card.pageUrl as string)}
              title="Open the full workflow in your browser"
              aria-label="Open the full workflow in your browser"
              className="ml-auto flex h-5 w-5 items-center justify-center rounded text-zinc-500 transition-colors hover:bg-surface hover:text-zinc-200"
            >
              <ExternalLinkIcon className="h-3.5 w-3.5" />
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}
