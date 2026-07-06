import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useLightboxStore } from '../store/lightboxStore'
import { CloseIcon } from './icons'

/**
 * Fullscreen media viewer, opened by double-clicking a node's image/video. Portaled to
 * document.body so it sits above the canvas and all panels. Click the backdrop or press Escape
 * to close.
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
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 p-8 backdrop-blur-sm"
      onClick={close}
    >
      {media.kind === 'video' ? (
        <video
          src={media.src}
          controls
          autoPlay
          onClick={(e) => e.stopPropagation()}
          className="max-h-full max-w-full rounded-lg shadow-2xl"
        />
      ) : (
        <img
          src={media.src}
          alt={media.name ?? ''}
          onClick={(e) => e.stopPropagation()}
          className="max-h-full max-w-full rounded-lg object-contain shadow-2xl"
        />
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
