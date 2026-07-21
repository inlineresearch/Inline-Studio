import { NodeToolbar, Position } from '@xyflow/react'
import { PlayIcon, StopIcon } from './NodeBadge'

/**
 * The single Run control for a selected graph, floated just above its **output node** (this node)
 * and tracking it on pan/zoom via React Flow's NodeToolbar. Shown only when the node is the selected
 * graph's run target; becomes Stop while the graph is generating. Each node type wires its own run /
 * cancel action (a Core node → runWorkflow, a fal frame → run), so this stays presentational.
 */
export function NodeRunToolbar({
  isTarget,
  busy,
  onRun,
  onStop,
  disabled = false,
  disabledReason,
}: {
  isTarget: boolean
  busy: boolean
  onRun: () => void
  onStop: () => void
  disabled?: boolean
  disabledReason?: string
}): React.JSX.Element {
  return (
    <NodeToolbar isVisible={isTarget} position={Position.Top} align="end" offset={12}>
      {busy ? (
        <button
          onClick={onStop}
          title="Interrupt generation"
          className="nodrag flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-[11px] font-medium text-zinc-200 shadow-lg hover:bg-black/40 hover:text-white"
        >
          <StopIcon className="h-3.5 w-3.5" />
          Stop
        </button>
      ) : (
        <button
          onClick={onRun}
          disabled={disabled}
          title={disabled ? disabledReason : 'Run this graph'}
          className="nodrag flex items-center gap-1.5 rounded-full bg-emerald-500 px-3 py-1 text-[11px] font-semibold text-black shadow-lg hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <PlayIcon className="h-3.5 w-3.5" />
          Run graph
        </button>
      )}
    </NodeToolbar>
  )
}
