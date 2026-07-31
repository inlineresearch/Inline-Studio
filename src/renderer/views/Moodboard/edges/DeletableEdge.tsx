import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  getSmoothStepPath,
  type EdgeProps,
} from '@xyflow/react'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useCanvasPrefsStore } from '../../../store/canvasPrefsStore'

/**
 * A connector that highlights when clicked and shows a ✕ button at its midpoint to
 * remove the link between two nodes. `data.functional` distinguishes the animated
 * output→preview wire (indigo) from purely-visual frame links (gray).
 *
 * The Studio and Trainer canvases keep separate stores, and React Flow builds edges from a type map
 * and cannot pass props. So the drawing lives in `DeletableEdgeBody`, which takes `disconnect`
 * explicitly, and each canvas wraps it with its own store. One implementation, two thin wrappers.
 */
export function DeletableEdge(props: EdgeProps): React.JSX.Element {
  return <DeletableEdgeBody {...props} disconnect={useMoodboardStore((s) => s.disconnect)} />
}

export function DeletableEdgeBody(
  props: EdgeProps & { disconnect: (id: string) => void | Promise<void> },
): React.JSX.Element {
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
    disconnect,
  } = props
  // Line style is a live canvas preference - every edge subscribes, so switching it in Settings
  // restyles all connectors at once. `angled` = cornered (smooth step); `wave` = curved (bezier).
  const edgeStyle = useCanvasPrefsStore((s) => s.edgeStyle)
  const geom = { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition }
  const [path, labelX, labelY] =
    edgeStyle === 'angled' ? getSmoothStepPath(geom) : getBezierPath(geom)
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
