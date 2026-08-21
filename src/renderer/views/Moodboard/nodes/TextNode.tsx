import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { studio } from '@/lib/studio'
import type { NodeProps } from '@xyflow/react'
import { NodeFrame } from './NodeFrame'
import { TextToolbar } from './TextToolbar'
import { useMoodboardStore } from '../../../store/moodboardStore'
import type { TextNodeData } from './nodeData'
import type { TextItemData } from '@shared/types'
import { useAutoCommit } from '../../../lib/useAutoCommit'

/**
 * Editable text item - floats bare on the canvas (no surface box), light-grey by
 * default. Double-click to edit; blur persists. Selecting it reveals a formatting
 * toolbar (colour / size / style / align / link). A linked, non-editing node opens
 * its URL in the browser on click. Resizing only changes the box - font size is set
 * from the toolbar, not from the container dimensions.
 */
export function TextNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { text } = data as TextNodeData
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const item = useMoodboardStore((s) => s.items.find((it) => it.id === id))
  const [editing, setEditing] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  // Latest item in a ref so fitHeight can stay stable yet read current dimensions.
  const itemRef = useRef(item)
  itemRef.current = item

  // Keep the node's height wrapped tightly around the text - both growing (e.g.
  // after pasting a long block) and shrinking, so the selectable/clickable box
  // never extends past the visible text into empty canvas. The editable div sizes
  // to its content (no `h-full`), so `scrollHeight` is the true text height; we add
  // the fixed NodeFrame chrome (p-1 padding + 1px border, top and bottom).
  const CHROME = 10
  const fitHeight = useCallback((): void => {
    const el = ref.current
    const it = itemRef.current
    if (!el || !it) return
    const needed = Math.max(32, el.scrollHeight + CHROME)
    // Programmatic auto-fit - don't pollute the undo history.
    if (Math.abs(needed - it.height) > 1) void updateItem(id, { height: needed }, false)
  }, [id, updateItem])

  const style: CSSProperties = {
    fontSize: text.fontSize,
    color: text.color,
    textAlign: text.align,
    fontWeight: text.bold ? 700 : 400,
    fontStyle: text.italic ? 'italic' : 'normal',
    textDecoration: text.underline || (text.link && !editing) ? 'underline' : 'none',
    cursor: text.link && !editing ? 'pointer' : undefined,
  }

  const save = (): void => {
    const next = ref.current?.innerText ?? text.text
    if (next !== text.text) void updateItem(id, { data: { text: { ...text, text: next } } })
  }
  const { schedule, flush } = useAutoCommit(save)
  const commit = (): void => {
    setEditing(false)
    flush()
  }

  const applyPatch = (patch: Partial<TextItemData>): void =>
    void updateItem(id, { data: { text: { ...text, ...patch } } })

  const openLink = (): void => {
    if (text.link && !editing) void studio().shell.openExternal(text.link)
  }

  // Refit when the text, font size/weight/style, or node width (wrapping) changes.
  useLayoutEffect(() => {
    fitHeight()
  }, [fitHeight, text.text, text.fontSize, text.bold, text.italic, item?.width, editing])

  return (
    <>
      {/* Sibling of the frame (not a child) so it escapes the node's overflow-hidden box. */}
      {selected && !editing && <TextToolbar text={text} onChange={applyPatch} />}
      <NodeFrame id={id} selected={!!selected} minHeight={32} transparent>
        <div
          ref={ref}
          style={style}
          className={`w-full whitespace-pre-wrap break-words px-1 outline-none ${
            editing ? 'nodrag cursor-text' : 'cursor-grab'
          }`}
          contentEditable={editing}
          suppressContentEditableWarning
          onDoubleClick={() => setEditing(true)}
          onClick={openLink}
          onInput={() => {
            fitHeight()
            schedule()
          }}
          onBlur={commit}
        >
          {text.text}
        </div>
      </NodeFrame>
    </>
  )
}
