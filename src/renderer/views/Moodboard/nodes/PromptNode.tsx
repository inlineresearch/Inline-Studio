import { useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { NodeFrame } from './NodeFrame'
import { NodeBadge, NodeBadgeRow } from './NodeBadge'

/**
 * A text-prompt node: a bare textarea whose output (a dot on the right) feeds a Generate node's
 * prompt input. The text is the single source of truth — the executor reads it at run time.
 */
export function PromptNode({ id, selected }: NodeProps): React.JSX.Element {
  const item = useMoodboardStore((s) => s.items.find((it) => it.id === id))
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const [text, setText] = useState<string>(() => (item?.data.promptText as string) ?? '')

  const commit = (): void => {
    if (!item) return
    void updateItem(id, { data: { ...item.data, promptText: text } })
  }

  return (
    <>
      {/* Label badge — floats above the node, outside its container. */}
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

      {/* Text output — connect this to a Generate node's prompt dot. */}
      <Handle
        type="source"
        id="out"
        position={Position.Right}
        className="!h-3 !w-3 !border-2 !border-surface !bg-amber-400"
      />
    </>
  )
}
