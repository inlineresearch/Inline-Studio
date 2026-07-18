import { takeWaveformPath } from '@shared/media'
import type { AssetKind } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { setFrameDragPayload, setMediaFileDragPayload, setOutputDragPayload } from '../../lib/dnd'
import { useMediaContextMenu } from '../../lib/mediaContextMenu'
import { VideoPreview } from '../../components/VideoPreview'
import { AudioPreview } from '../../components/AudioPreview'

/** One generated output in the Outputs gallery, normalized across frame takes and Core-node renders. */
export interface OutputTile {
  /** Take id (used as the drag's take pin and as the React key). */
  id: string
  /** Project-relative media path. */
  filePath: string
  kind: AssetKind
  /** Producing frame or node name, shown under the thumb. */
  label: string
  /**
   * Producing frame id - set for frame takes, which drag as a frame/output payload (fed as a flow
   * link). Null for Core-node outputs, which have no frame, so they drag as a raw media file (the
   * drop target imports it into the Library first).
   */
  frameId: string | null
}

/**
 * A generated output (take) tile. Drag it onto a generation node to feed it as an input, or onto the
 * canvas to make a new frame. Frame takes carry their frame/output payload; Core-node outputs (no
 * frame) carry a media-file payload - the drop target imports the file into the Library first.
 */
export function OutputThumb({ tile }: { tile: OutputTile }): React.JSX.Element {
  const { id, filePath, kind, label, frameId } = tile
  const src = resolveMedia(filePath)
  const onContextMenu = useMediaContextMenu()
  const onDragStart = (e: React.DragEvent): void => {
    if (frameId !== null) {
      // Frame payload → drop on a node feeds it as input; output payload → drop on canvas creates a
      // new frame fed by this output. The take id pins the exact image dragged.
      setFrameDragPayload(e.dataTransfer, frameId)
      setOutputDragPayload(e.dataTransfer, frameId, id)
    } else {
      // Core-node output: no frame, so carry the raw file for the drop target to import.
      setMediaFileDragPayload(e.dataTransfer, { filePath, kind, name: `${label}.${extFor(kind)}` })
    }
  }
  return (
    <div
      draggable
      onDragStart={onDragStart}
      onContextMenu={(e) => onContextMenu(e, { src, name: label, kind })}
      title={`${label} - drag onto a node to use as input, or onto the canvas to make a frame`}
      className="flex w-full cursor-grab flex-col overflow-hidden rounded-md border border-border hover:border-zinc-600"
    >
      <div className="flex aspect-video items-center justify-center bg-black/40">
        {kind === 'image' && <img src={src} alt={label} className="h-full w-full object-cover" />}
        {kind === 'video' && <VideoPreview src={src} className="h-full w-full object-cover" />}
        {kind === 'audio' && (
          <AudioPreview
            src={src}
            waveformUrl={resolveMedia(takeWaveformPath(id))}
            className="h-full w-full"
          />
        )}
      </div>
      <span className="truncate px-1.5 py-1 text-[11px] text-emerald-300/80">{label}</span>
    </div>
  )
}

/** A default file extension per media kind, for naming an imported Core-node output. */
function extFor(kind: AssetKind): string {
  return kind === 'video' ? 'mp4' : kind === 'audio' ? 'mp3' : 'png'
}
