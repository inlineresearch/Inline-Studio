import { useState } from 'react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { ChevronDownIcon, XIcon } from './NodeBadge'

interface LayerNodeData extends Record<string, unknown> {
  name: string
  color?: string
}

const DEFAULT_COLOR = '#93c5fd' // blue-300

/** Ten light preset layer colors for the header dropdown. */
const LAYER_COLORS = [
  '#93c5fd', // blue-300
  '#cbd5e1', // slate-300
  '#fca5a5', // red-300
  '#fdba74', // orange-300
  '#fcd34d', // amber-300
  '#86efac', // green-300
  '#5eead4', // teal-300
  '#a5b4fc', // indigo-300
  '#d8b4fe', // purple-300
  '#f9a8d4', // pink-300
]

/**
 * A resizable, renamable group container with a color. Frames dropped inside become
 * its React Flow children and move with it. Only the title bar (.drag-handle) drags
 * the layer; the body ignores pointer events so the frames/links inside stay usable.
 */
export function LayerNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { name: rawName, color: rawColor } = data as LayerNodeData
  const name = rawName || 'Layer'
  const color = rawColor || DEFAULT_COLOR
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const deleteItem = useMoodboardStore((s) => s.deleteItem)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(name)
  const [pickerOpen, setPickerOpen] = useState(false)

  const commitName = (): void => {
    setEditing(false)
    const next = draft.trim() || 'Layer'
    if (next !== name) void updateItem(id, { data: { name: next, color: rawColor } })
  }

  const pickColor = (c: string): void => {
    setPickerOpen(false)
    if (c !== rawColor) void updateItem(id, { data: { name, color: c } })
  }

  return (
    <>
      <NodeResizer
        isVisible={selected}
        minWidth={220}
        minHeight={160}
        lineClassName="!border-accent"
        handleClassName="!h-2 !w-2 !rounded-sm !bg-accent !border-white"
        onResizeEnd={(_e, p) => void updateItem(id, { width: p.width, height: p.height })}
      />
      {/* Body ignores pointer events so clicks fall through to the frames and
          connectors drawn inside the layer; only the title bar is interactive. */}
      <div
        className="pointer-events-none flex h-full w-full flex-col overflow-hidden rounded-lg border-2"
        style={{ borderColor: color, backgroundColor: `${color}14` }}
      >
        <div className="drag-handle pointer-events-auto flex cursor-grab items-center gap-2 border-b border-border bg-panel px-2 py-1">
          <span
            className="rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-zinc-900"
            style={{ backgroundColor: color }}
          >
            Layer
          </span>
          {editing ? (
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitName}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitName()
                if (e.key === 'Escape') setEditing(false)
              }}
              className="nodrag min-w-0 flex-1 bg-transparent text-xs font-medium text-zinc-100 outline-none"
            />
          ) : (
            <span
              onDoubleClick={() => {
                setDraft(name)
                setEditing(true)
              }}
              title="Double-click to rename"
              className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-200"
            >
              {name}
            </span>
          )}

          {selected && (
            <button
              onClick={() => void deleteItem(id)}
              title="Delete layer"
              className="nodrag nopan flex shrink-0 items-center px-0.5 text-zinc-400 hover:text-red-400"
            >
              <XIcon className="h-3 w-3" />
            </button>
          )}

          {/* Color picker: a circle of the current color + a caret. */}
          <div className="relative shrink-0">
            <button
              onClick={() => setPickerOpen((o) => !o)}
              title="Layer color"
              className="nodrag nopan flex items-center gap-0.5 rounded px-0.5 py-0.5 hover:bg-surface"
            >
              <span
                className="h-3.5 w-3.5 rounded-full border border-white/40"
                style={{ backgroundColor: color }}
              />
              <ChevronDownIcon className="h-3 w-3 text-zinc-400" />
            </button>
            {pickerOpen && (
              <div
                className="nodrag nopan absolute right-0 top-full z-50 mt-1 grid grid-cols-5 gap-1.5 rounded-md border border-border bg-panel p-2 shadow-lg"
                style={{ width: 124 }}
              >
                {LAYER_COLORS.map((c) => (
                  <button
                    key={c}
                    onClick={() => pickColor(c)}
                    title={c}
                    className={`h-4 w-4 rounded-full border ${
                      c === color ? 'border-white' : 'border-white/30'
                    } hover:scale-110`}
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
