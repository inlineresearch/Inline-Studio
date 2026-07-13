import { takeWaveformPath } from '@shared/media'
import type { Take } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { setFrameDragPayload, setOutputDragPayload } from '../../lib/dnd'
import { useMediaContextMenu } from '../../lib/mediaContextMenu'
import { VideoPreview } from '../../components/VideoPreview'
import { AudioPreview } from '../../components/AudioPreview'

/**
 * A generated output (take) tile. Drag it onto a generation node to feed it as an input (it
 * carries its producing frame id — the flow-link resolves to this take at generate time).
 */
export function OutputThumb({
  take,
  frameName,
}: {
  take: Take
  frameName: string
}): React.JSX.Element {
  const src = resolveMedia(take.filePath)
  const onContextMenu = useMediaContextMenu()
  return (
    <div
      draggable
      onDragStart={(e) => {
        // Frame payload → drop on a node feeds it as input; output payload → drop on canvas
        // creates a new frame fed by this output. The take id pins the exact image dragged.
        setFrameDragPayload(e.dataTransfer, take.frameId)
        setOutputDragPayload(e.dataTransfer, take.frameId, take.id)
      }}
      onContextMenu={(e) => onContextMenu(e, { src, name: frameName, kind: take.kind })}
      title={`${frameName} — drag onto a node to use as input, or onto the canvas to make a frame`}
      className="flex w-full cursor-grab flex-col overflow-hidden rounded-md border border-border hover:border-zinc-600"
    >
      <div className="flex aspect-video items-center justify-center bg-black/40">
        {take.kind === 'image' && (
          <img src={src} alt={frameName} className="h-full w-full object-cover" />
        )}
        {take.kind === 'video' && <VideoPreview src={src} className="h-full w-full object-cover" />}
        {take.kind === 'audio' && (
          <AudioPreview
            src={src}
            waveformUrl={resolveMedia(takeWaveformPath(take.id))}
            className="h-full w-full"
          />
        )}
      </div>
      <span className="truncate px-1.5 py-1 text-[11px] text-emerald-300/80">{frameName}</span>
    </div>
  )
}
