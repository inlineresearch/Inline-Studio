import { VideoPreview } from '../../../components/VideoPreview'
import { resolveMedia } from '@/lib/media'

/** What a Core node's take can be. Mirrors `MediaKind` on the engine side. */
export type OutputKind = 'image' | 'video' | 'audio'

interface Props {
  filePath: string
  kind: OutputKind
  /** Shown as the lightbox caption and the image alt text. */
  name: string
  /** Audio is absent on purpose: there is nothing to zoom into, so it never expands. */
  onExpand?: (kind: 'image' | 'video') => void
  className?: string
}

/**
 * One Core take, rendered for whatever media kind it is.
 *
 * Extracted so the main preview and the take-history strip stay in step: before this, both branched
 * on `kind === 'image'` separately and anything else fell through to a text placeholder, so the
 * first video model would have shown the word "video" instead of the video.
 */
export function CoreOutputPreview({
  filePath,
  kind,
  name,
  onExpand,
  className,
}: Props): React.JSX.Element {
  const src = resolveMedia(filePath)
  const shared = className ?? 'h-full w-full object-cover'

  if (kind === 'video') {
    return (
      <VideoPreview
        src={src}
        className={`${shared} cursor-zoom-in`}
        onDoubleClick={() => onExpand?.('video')}
      />
    )
  }

  if (kind === 'audio') {
    // H3 and anything else that generates a soundtrack expose it on its own port. There is nothing
    // to show, so give it something to play rather than a label saying "audio".
    return (
      <div className="flex h-full w-full items-center justify-center bg-black px-3">
        <audio src={src} controls className="nodrag w-full max-w-[240px]" />
      </div>
    )
  }

  return (
    <img
      src={src}
      alt={name}
      title="Double-click to expand"
      onDoubleClick={() => onExpand?.('image')}
      className={`${shared} cursor-zoom-in`}
    />
  )
}

/** The same take at thumbnail size, for the take-history strip. */
export function CoreOutputThumb({
  filePath,
  kind,
}: {
  filePath: string
  kind: OutputKind
}): React.JSX.Element {
  const src = resolveMedia(filePath)
  if (kind === 'image') {
    return <img src={src} alt="" className="h-full w-full object-cover" />
  }
  if (kind === 'video') {
    // Muted and unplayed: a strip of autoplaying clips would fight for attention with the node's
    // own preview. `preload="metadata"` is enough for the browser to paint the first frame.
    return (
      <video
        src={src}
        muted
        playsInline
        preload="metadata"
        className="h-full w-full object-cover"
      />
    )
  }
  return (
    <div className="flex h-full w-full items-center justify-center bg-black">
      <WaveformIcon className="h-4 w-4 text-zinc-500" />
    </div>
  )
}

function WaveformIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      className={className}
      aria-hidden
    >
      <path d="M3 12h2M8 7v10M12 4v16M16 8v8M21 12h-2" />
    </svg>
  )
}
