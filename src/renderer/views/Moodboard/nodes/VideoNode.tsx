import type { NodeProps } from '@xyflow/react'
import { NodeFrame } from './NodeFrame'
import { NodeBadge, NodeBadgeRow, VideoGlyph } from './NodeBadge'
import type { AssetNodeData } from './nodeData'
import { useMediaContextMenu } from '../../../lib/mediaContextMenu'
import { useLightboxStore } from '../../../store/lightboxStore'

export function VideoNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { src, name } = data as AssetNodeData
  const onContextMenu = useMediaContextMenu()
  const openLightbox = useLightboxStore((s) => s.open)
  // `nodrag` lets the player controls work without the canvas dragging the node.
  return (
    <>
      <NodeBadgeRow>
        <NodeBadge icon={<VideoGlyph />} title={name}>
          {name}
        </NodeBadge>
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect>
        <video
          src={src}
          controls
          onContextMenu={(e) => onContextMenu(e, { src, name, kind: 'video' })}
          onDoubleClick={() => openLightbox({ src, kind: 'video', name })}
          className="nodrag h-full w-full bg-black object-contain"
        />
      </NodeFrame>
    </>
  )
}
