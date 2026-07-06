import type { NodeProps } from '@xyflow/react'
import { NodeFrame } from './NodeFrame'
import { ImageGlyph, NodeBadge, NodeBadgeRow } from './NodeBadge'
import type { AssetNodeData } from './nodeData'
import { useMediaContextMenu } from '../../../lib/mediaContextMenu'
import { useLightboxStore } from '../../../store/lightboxStore'

export function ImageNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { src, name } = data as AssetNodeData
  const onContextMenu = useMediaContextMenu()
  const openLightbox = useLightboxStore((s) => s.open)
  return (
    <>
      <NodeBadgeRow>
        <NodeBadge icon={<ImageGlyph />} title={name}>
          {name}
        </NodeBadge>
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect>
        <img
          src={src}
          alt={name}
          draggable={false}
          onContextMenu={(e) => onContextMenu(e, { src, name, kind: 'image' })}
          onDoubleClick={() => openLightbox({ src, kind: 'image', name })}
          className="h-full w-full object-cover"
        />
      </NodeFrame>
    </>
  )
}
