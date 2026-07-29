import type { ReactNode } from 'react'
import { NodeResizer } from '@xyflow/react'
import { useBoardActions } from './boardActions'
import { XIcon } from './NodeBadge'

/**
 * Shared chrome for every moodboard node: a fill container, a resize handle +
 * delete button when selected. Resize persists on end; delete removes the item.
 */
/** New dimensions reported by the resize handle. */
export interface ResizeSize {
  width: number
  height: number
}

export function NodeFrame({
  id,
  selected,
  minWidth = 80,
  minHeight = 40,
  padded = true,
  transparent = false,
  subtleSelect = false,
  running = false,
  overflowVisible = false,
  onResizeStart,
  onResize,
  onResizeEnd,
  children,
}: {
  id: string
  selected: boolean
  minWidth?: number
  minHeight?: number
  padded?: boolean
  /** Drop the surface box (border + background) - used by text, which floats bare on the canvas. */
  transparent?: boolean
  /** Use a soft light-grey selection outline instead of the loud accent (for media-heavy nodes). */
  subtleSelect?: boolean
  /** This node is currently executing - show a green border so it's clear which node is running. */
  running?: boolean
  /** Let content overflow the card (e.g. a dropdown that spills past the frame). Off by default. */
  overflowVisible?: boolean
  /** Resize hooks. When `onResizeEnd` is given it replaces the default width/height persist. */
  onResizeStart?: (size: ResizeSize) => void
  onResize?: (size: ResizeSize) => void
  onResizeEnd?: (size: ResizeSize) => void
  children: ReactNode
}): React.JSX.Element {
  const { updateItem, deleteItem } = useBoardActions()

  // A running node's green border wins over the selection colour so it's always clear what's executing.
  const selBorder = subtleSelect ? 'border-zinc-600' : 'border-accent'
  const activeBorder = running ? 'border-emerald-400' : selected ? selBorder : 'border-border'
  const box = transparent
    ? `bg-transparent ${running ? 'border border-emerald-400' : selected ? `border border-dashed ${subtleSelect ? 'border-zinc-600/70' : 'border-accent/60'}` : 'border border-transparent'}`
    : `border bg-surface ${activeBorder}`
  const ring = running ? 'ring-2 ring-emerald-400/30' : ''

  return (
    <>
      <NodeResizer
        isVisible={selected}
        minWidth={minWidth}
        minHeight={minHeight}
        lineClassName={subtleSelect ? '!border-zinc-600' : '!border-accent'}
        handleClassName={subtleSelect ? '!bg-zinc-500 !border-white' : '!bg-accent !border-white'}
        onResizeStart={(_e, p) => onResizeStart?.({ width: p.width, height: p.height })}
        onResize={(_e, p) => onResize?.({ width: p.width, height: p.height })}
        onResizeEnd={(_e, p) => {
          const size = { width: p.width, height: p.height }
          if (onResizeEnd) onResizeEnd(size)
          else void updateItem(id, size)
        }}
      />
      <div
        className={`h-full w-full rounded-md ${overflowVisible ? 'overflow-visible' : 'overflow-hidden'} ${box} ${ring} ${padded ? 'p-1' : ''}`}
      >
        {children}
      </div>
      {selected && (
        <button
          onClick={() => void deleteItem(id)}
          title="Delete"
          className="absolute -right-2 -top-2 z-10 flex h-5 w-5 items-center justify-center rounded-full bg-black/80 text-zinc-200 hover:text-red-400"
        >
          <XIcon className="h-3 w-3" />
        </button>
      )}
    </>
  )
}
