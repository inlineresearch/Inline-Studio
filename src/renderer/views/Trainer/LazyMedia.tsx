/** Media that only loads once it is on screen. */
import { useRef } from 'react'
import type { Asset } from '@shared/types'
import { useLightboxStore } from '../../store/lightboxStore'
import { useSeen } from './useSeen'
import { resolveMedia } from '@/lib/media'

/**
 * One asset's preview.
 *
 * A dataset is hundreds of clips, and a `<video preload="metadata">` per row means hundreds of
 * simultaneous range requests: enough to hang the tab and saturate the server. Nothing is requested
 * until the row is close to view.
 */
export function LazyMedia({
  asset,
  compare,
  className = '',
}: {
  asset: Asset
  /** The other half of a pair, opened beside this one on double click. */
  compare?: Asset
  className?: string
}): React.JSX.Element {
  const [ref, seen] = useSeen<HTMLDivElement>()
  const video = useRef<HTMLVideoElement>(null)
  const openLightbox = useLightboxStore((s) => s.open)
  const src = resolveMedia(asset.thumbPath ?? asset.filePath)

  const asMedia = (a: Asset) => ({
    // previewPath is the transcode for codecs Chromium cannot decode; the original otherwise.
    src: resolveMedia(a.previewPath ?? a.filePath),
    kind: a.kind === 'video' ? ('video' as const) : ('image' as const),
    name: a.name,
  })

  return (
    <div
      ref={ref}
      className={`h-full w-full bg-black ${className}`}
      title="Double click to view"
      onDoubleClick={() =>
        openLightbox({ ...asMedia(asset), compare: compare ? asMedia(compare) : undefined })
      }
      onMouseEnter={() => void video.current?.play().catch(() => {})}
      onMouseLeave={() => {
        const v = video.current
        if (!v) return
        v.pause()
        v.currentTime = 0.1
      }}
    >
      {!seen ? null : asset.kind === 'video' && !asset.thumbPath ? (
        // Poster generation is deferred for clips, so an <img> at an mp4 renders broken. `#t=`
        // seeks the browser to a frame without playing it; hovering is what starts playback.
        <video
          ref={video}
          src={`${src}#t=0.1`}
          muted
          loop
          playsInline
          preload="metadata"
          className="h-full w-full object-cover"
        />
      ) : (
        <img src={src} alt={asset.name} loading="lazy" className="h-full w-full object-cover" />
      )}
    </div>
  )
}
