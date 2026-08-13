import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useLightboxStore } from '../store/lightboxStore'
import { CloseIcon } from './icons'

/**
 * Fullscreen media viewer, opened by double-clicking a node's or dataset tile's image/video.
 * Portaled to document.body so it sits above the canvas and all panels. Its z-index must stay
 * above every modal (the Captioning modal is z-200), since it can be opened from inside one.
 * Click the backdrop or press Escape to close.
 */
export function MediaLightbox(): React.JSX.Element | null {
  const media = useLightboxStore((s) => s.media)
  const close = useLightboxStore((s) => s.close)

  useEffect(() => {
    if (!media) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [media, close])

  if (!media) return null

  return createPortal(
    <div
      className="fixed inset-0 z-[300] flex items-center justify-center bg-black/85 p-8 backdrop-blur-sm"
      onClick={close}
    >
      {media.compare ? (
        <div className="flex h-full w-full items-center justify-center gap-3">
          <Pane media={media} half />
          <Pane media={media.compare} half />
        </div>
      ) : (
        <Pane media={media} />
      )}
      <button
        onClick={close}
        aria-label="Close"
        title="Close (Esc)"
        className="absolute right-4 top-4 flex h-9 w-9 items-center justify-center rounded-full bg-black/60 text-zinc-200 hover:bg-black/80 hover:text-white"
      >
        <CloseIcon className="h-5 w-5" />
      </button>
    </div>,
    document.body,
  )
}

function Pane({
  media,
  half,
}: {
  media: { src: string; kind: 'image' | 'video'; name?: string }
  /** One of a side-by-side pair, so it may claim at most half the width. */
  half?: boolean
}): React.JSX.Element {
  const stop = (e: React.MouseEvent): void => e.stopPropagation()
  // Sized against the viewport rather than the parent: a video reports its intrinsic dimensions
  // before any layout applies, so `max-w-full` alone lets a 1920-wide clip push past the screen,
  // and two of them in a row overflow it outright. The 4rem is the backdrop's own padding.
  const size = half
    ? 'max-h-[calc(100vh-4rem)] max-w-[calc(50vw-2.5rem)]'
    : 'max-h-[calc(100vh-4rem)] max-w-[calc(100vw-4rem)]'
  const className = `${size} rounded-lg object-contain shadow-2xl`
  return media.kind === 'video' ? (
    <video src={media.src} controls autoPlay loop muted onClick={stop} className={className} />
  ) : (
    <img src={media.src} alt={media.name ?? ''} onClick={stop} className={className} />
  )
}
