import { useEffect, useRef, useState } from 'react'
import { VideoPreview } from '../../../components/VideoPreview'
import { needsBlurFill } from '../../../lib/loaderFit'

/** Downscaled hard so the blurred backdrop costs a thumbnail, not a second full-res decode. */
const FRAME_GRAB_W = 64

// A node's media, plus a blurred copy of itself filling whatever a contain fit leaves over.
export function MediaBody({
  src,
  kind,
  poster,
  fit = 'contain',
  onContextMenu,
  onDoubleClick,
  onAspect,
}: {
  src: string
  kind: 'image' | 'video'
  poster?: string
  fit?: 'contain' | 'cover'
  onContextMenu?: (e: React.MouseEvent) => void
  onDoubleClick?: () => void
  /** Fires with the media's intrinsic width/height ratio once it decodes. */
  onAspect?: (aspect: number) => void
}): React.JSX.Element {
  const boxRef = useRef<HTMLDivElement>(null)
  const [aspect, setAspect] = useState<number | null>(null)
  const [boxAspect, setBoxAspect] = useState<number | null>(null)
  // A video frame can't be a CSS background, so one grab stands in as the backdrop's still.
  const [videoStill, setVideoStill] = useState<string | null>(null)

  useEffect(() => {
    setAspect(null)
    setVideoStill(null)
  }, [src])

  useEffect(() => {
    const box = boxRef.current
    if (!box || fit !== 'contain') return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setBoxAspect(width > 0 && height > 0 ? width / height : null)
    })
    ro.observe(box)
    return () => ro.disconnect()
  }, [fit])

  const takeAspect = (w: number, h: number): void => {
    if (!w || !h) return
    setAspect(w / h)
    onAspect?.(w / h)
  }

  const grabStill = (v: HTMLVideoElement): void => {
    if (fit !== 'contain' || videoStill || !v.videoWidth || !v.videoHeight) return
    const canvas = document.createElement('canvas')
    canvas.width = FRAME_GRAB_W
    canvas.height = Math.max(1, Math.round((FRAME_GRAB_W * v.videoHeight) / v.videoWidth))
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    try {
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
      setVideoStill(canvas.toDataURL('image/jpeg', 0.6))
    } catch {
      // A frame we can't read just means no backdrop; the plain black body still works.
    }
  }

  const fitClass = fit === 'cover' ? 'object-cover' : 'object-contain'
  const backdrop = kind === 'video' ? (videoStill ?? poster) : src
  const showBlur =
    fit === 'contain' && !!backdrop && aspect != null && boxAspect != null
      ? needsBlurFill(aspect, boxAspect)
      : false

  return (
    <div ref={boxRef} className="relative h-full w-full overflow-hidden">
      {showBlur && (
        <>
          <div
            aria-hidden
            style={{ backgroundImage: `url("${backdrop}")` }}
            className="absolute inset-0 scale-125 bg-cover bg-center blur-2xl"
          />
          <div aria-hidden className="absolute inset-0 bg-black/50" />
        </>
      )}
      {kind === 'video' ? (
        <VideoPreview
          src={src}
          poster={poster}
          onLoadedMetadata={(e) =>
            takeAspect(e.currentTarget.videoWidth, e.currentTarget.videoHeight)
          }
          onLoadedData={(e) => grabStill(e.currentTarget)}
          onContextMenu={onContextMenu}
          onDoubleClick={onDoubleClick}
          className={`relative h-full w-full ${fitClass}`}
        />
      ) : (
        <img
          src={src}
          alt=""
          onLoad={(e) => takeAspect(e.currentTarget.naturalWidth, e.currentTarget.naturalHeight)}
          onContextMenu={onContextMenu}
          onDoubleClick={onDoubleClick}
          className={`relative h-full w-full ${fitClass}`}
        />
      )}
    </div>
  )
}
