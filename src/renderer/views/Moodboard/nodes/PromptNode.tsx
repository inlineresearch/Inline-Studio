import { useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { NodeFrame } from './NodeFrame'
import { NodeBadge, NodeBadgeRow } from './NodeBadge'

/**
 * A text-prompt node: a bare textarea whose output (a dot on the right) feeds a Generate node's
 * prompt input. The text is the single source of truth - the executor reads it at run time.
 */
export function PromptNode({ id, selected }: NodeProps): React.JSX.Element {
  const item = useMoodboardStore((s) => s.items.find((it) => it.id === id))
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const stored = (item?.data.promptText as string | undefined) ?? ''
  const [text, setText] = useState<string>(stored)
  // Re-seed when the stored text changes underneath us. The node mounts as soon as the item is
  // created, so anything that writes the text a moment later (a getting-started card, a recipe
  // rebuilt from a dropped image, an undo) would otherwise leave the textarea showing the empty
  // value it mounted with. Adjusting state during render rather than in an effect is React's own
  // guidance for this, and avoids a frame of stale text.
  const [seeded, setSeeded] = useState(stored)
  if (seeded !== stored) {
    setSeeded(stored)
    setText(stored)
  }

  const commit = (): void => {
    if (!item) return
    void updateItem(id, { data: { ...item.data, promptText: text } })
  }

  return (
    <>
      {/* Label badge - floats above the node, outside its container. */}
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<span className="text-sm font-bold leading-none text-zinc-400">T</span>}>
          Prompt
        </NodeBadge>
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={160}
        minHeight={80}
        padded={false}
        subtleSelect
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          spellCheck={false}
          placeholder="Describe what to generate…"
          className="nodrag h-full w-full resize-none bg-transparent p-2.5 text-xs leading-snug text-zinc-100 outline-none placeholder:text-zinc-600"
        />
      </NodeFrame>

      {/* Text output - connect this to a Generate node's prompt dot. */}
      <Handle
        type="source"
        id="out"
        position={Position.Right}
        className="!h-3 !w-3 !border-2 !border-surface !bg-amber-400"
      />
    </>
  )
}
