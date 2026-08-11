import { useEffect, useRef, useState } from 'react'
import { NodeToolbar, Position } from '@xyflow/react'
import { ChevronDownIcon, PlayIcon, StopIcon } from './NodeBadge'

export interface RunMenuItem {
  label: string
  onClick: () => void
}

/**
 * The single Run control for a selected graph, floated just above its **output node** (this node)
 * and tracking it on pan/zoom via React Flow's NodeToolbar. Shown only when the node is the selected
 * graph's run target; becomes Stop while the graph is generating. Each node type wires its own run /
 * cancel action (a Core node → runWorkflow, a fal frame → run), so this stays presentational.
 *
 * `menuItems` adds a caret beside Run for graph-level actions. It is optional because the Trainer
 * reuses this control for a training job, where copying a graph makes no sense.
 */
export function NodeRunToolbar({
  isTarget,
  busy,
  onRun,
  onStop,
  disabled = false,
  disabledReason,
  runLabel = 'Run graph',
  stopLabel = 'Stop',
  menuItems,
  menuNote,
}: {
  isTarget: boolean
  busy: boolean
  onRun: () => void
  onStop: () => void
  disabled?: boolean
  disabledReason?: string
  /** Override the chip labels (the Trainer node runs/stops a training job, not a graph). */
  runLabel?: string
  stopLabel?: string
  menuItems?: RunMenuItem[]
  /** Shown under the menu, e.g. that this graph holds node types an import cannot rebuild. */
  menuNote?: string
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent): void => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  // Close the menu when the toolbar goes away, or it reopens with the next selection.
  useEffect(() => {
    if (!isTarget) setOpen(false)
  }, [isTarget])

  return (
    <NodeToolbar isVisible={isTarget} position={Position.Top} align="end" offset={12}>
      <div ref={rootRef} className="relative flex items-center gap-1">
        {busy ? (
          <button
            data-run-toolbar
            onClick={onStop}
            title={stopLabel}
            className="nodrag flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-[11px] font-medium text-zinc-200 shadow-lg hover:bg-black/40 hover:text-white"
          >
            <StopIcon className="h-3.5 w-3.5" />
            {stopLabel}
          </button>
        ) : (
          <button
            data-run-toolbar
            onClick={onRun}
            disabled={disabled}
            title={disabled ? disabledReason : runLabel}
            className="nodrag flex items-center gap-1.5 rounded-full bg-emerald-500 px-3 py-1 text-[11px] font-semibold text-black shadow-lg hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <PlayIcon className="h-3.5 w-3.5" />
            {runLabel}
          </button>
        )}

        {menuItems && menuItems.length > 0 && (
          <button
            data-run-toolbar
            onClick={() => setOpen((o) => !o)}
            title="Graph actions"
            aria-label="Graph actions"
            aria-expanded={open}
            className="nodrag flex h-[22px] w-6 items-center justify-center rounded-full border border-border bg-surface text-zinc-300 shadow-lg hover:bg-black/40 hover:text-white"
          >
            <ChevronDownIcon className="h-3.5 w-3.5" />
          </button>
        )}

        {open && menuItems && (
          <div className="nodrag nowheel absolute right-0 top-full z-50 mt-1 w-52 overflow-hidden rounded-md border border-border bg-panel shadow-xl">
            {menuItems.map((item) => (
              <button
                key={item.label}
                onClick={() => {
                  setOpen(false)
                  item.onClick()
                }}
                className="block w-full px-3 py-1.5 text-left text-[11px] text-zinc-200 hover:bg-black/30"
              >
                {item.label}
              </button>
            ))}
            {menuNote && (
              <p className="border-t border-border px-3 py-1.5 text-[10px] leading-snug text-amber-400/80">
                {menuNote}
              </p>
            )}
          </div>
        )}
      </div>
    </NodeToolbar>
  )
}
