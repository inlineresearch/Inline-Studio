import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react'
import { useMoodboardStore } from '../../../store/moodboardStore'

/**
 * A connector that highlights when clicked and shows a ✕ button at its midpoint to
 * remove the link between two nodes. `data.functional` distinguishes the animated
 * output→preview wire (indigo) from purely-visual frame links (gray).
 */
export function DeletableEdge(props: EdgeProps): React.JSX.Element {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    markerEnd,
    selected,
    data,
  } = props
  const disconnect = useMoodboardStore((s) => s.disconnect)
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  })
  const edgeData = data as { functional?: boolean; color?: string; kindColor?: string } | undefined
  const functional = edgeData?.functional ?? false
  // An engine wire (typed Core port) always shows its dot's kind color; selection just thickens it.
  // Other links keep the rose-on-select / lime-functional / level-color scheme.
  const kindColor = edgeData?.kindColor
  const stroke = kindColor
    ? kindColor
    : selected
      ? '#fb7185'
      : functional
        ? '#DCE775'
        : (edgeData?.color ?? '#52525b')

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        interactionWidth={24}
        style={{ stroke, strokeWidth: selected ? 3 : 2 }}
      />
      {selected && (
        <EdgeLabelRenderer>
          <button
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'all',
              // Sit above the nodes so the midpoint button is always clickable,
              // even when the connector's midpoint falls over a frame.
              zIndex: 1000,
            }}
            className="nodrag nopan flex h-3.5 w-3.5 items-center justify-center rounded-full bg-rose-500 text-[8px] leading-none text-white shadow-md hover:bg-rose-400"
            title="Delete connector"
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation()
              void disconnect(id)
            }}
          >
            ✕
          </button>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
