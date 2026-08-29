import { useRef } from 'react'

/** How many times an autoplaying preview loops before it stops on its last frame. */
const MAX_PLAYS = 2

/**
 * A muted video preview that autoplays `MAX_PLAYS` times then stops on its last
 * frame. Hovering replays it once more each time. Used for video assets/inputs so
 * they preview in motion without running forever. `src` should be Chromium-playable.
 */
export function VideoPreview({
  src,
  poster,
  className,
  onLoadedMetadata,
  onLoadedData,
  onContextMenu,
  onDoubleClick,
}: {
  src: string
  poster?: string
  className?: string
  onLoadedMetadata?: React.ReactEventHandler<HTMLVideoElement>
  onLoadedData?: React.ReactEventHandler<HTMLVideoElement>
  onContextMenu?: React.MouseEventHandler<HTMLVideoElement>
  onDoubleClick?: React.MouseEventHandler<HTMLVideoElement>
}): React.JSX.Element {
  const ref = useRef<HTMLVideoElement>(null)
  // Plays remaining in the current run (the initial autoplay starts at MAX_PLAYS).
  const playsLeft = useRef(MAX_PLAYS)

  const onEnded = (): void => {
    const v = ref.current
    if (!v) return
    playsLeft.current -= 1
    if (playsLeft.current > 0) {
      v.currentTime = 0
      void v.play().catch(() => {})
    }
  }

  // Each hover plays the clip through one more time.
  const onMouseEnter = (): void => {
    const v = ref.current
    if (!v) return
    playsLeft.current = 1
    v.currentTime = 0
    void v.play().catch(() => {})
  }

  return (
    <video
      ref={ref}
      src={src}
      poster={poster}
      muted
      autoPlay
      playsInline
      preload="metadata"
      onEnded={onEnded}
      onMouseEnter={onMouseEnter}
      onLoadedMetadata={onLoadedMetadata}
      onLoadedData={onLoadedData}
      onContextMenu={onContextMenu}
      onDoubleClick={onDoubleClick}
      className={className}
    />
  )
}
