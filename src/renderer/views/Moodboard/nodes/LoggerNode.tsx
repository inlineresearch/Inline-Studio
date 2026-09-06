/** Shows the lines whatever run is wired into it is streaming, newest last. */
import { useEffect, useRef, useState } from 'react'
import { type NodeProps } from '@xyflow/react'
import { useLogStore } from '../../../store/logStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { copyText } from '../../../lib/clipboard'
import { NodeFrame } from './NodeFrame'
import { PortHandle } from './PortHandle'
import { topStyle } from './nodeSize'
import { DownloadIcon, NodeBadge, NodeBadgeRow, TerminalIcon } from './NodeBadge'
import { CopyIcon } from '../../../components/icons'

/** The run id of whatever feeds this node, the way the loss graph finds its own. */
function wiredRunId(
  itemId: string,
  connectors: { fromItemId: string; toItemId: string }[],
  items: { id: string; data: { runId?: string | null } }[],
): string | null {
  const incoming = connectors.find((c) => c.toItemId === itemId)
  if (!incoming) return null
  return items.find((i) => i.id === incoming.fromItemId)?.data.runId ?? null
}

/** Rendered tail depth. The store holds far more; this bounds the DOM, not the buffer. */
const SHOWN = 200

export function LoggerNode({ id, selected }: NodeProps): React.JSX.Element {
  const connectors = useMoodboardStore((s) => s.connectors)
  const items = useMoodboardStore((s) => s.items)
  const runId = wiredRunId(id, connectors, items)
  const lines = useLogStore((s) => (runId ? s.linesByRun[runId] : undefined)) ?? []
  const clear = useLogStore((s) => s.clear)
  const [copied, setCopied] = useState(false)

  // Follow the tail, but only while the user is already at the bottom - otherwise scrolling back
  // through the log would be yanked to the end every time a new line streams in.
  const logRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)
  useEffect(() => {
    const el = logRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [lines.length])

  const copy = async (): Promise<void> => {
    await copyText(lines.join('\n'))
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  const download = (): void => {
    const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/plain' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `${runId ?? 'log'}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<TerminalIcon className="h-3.5 w-3.5 shrink-0 text-zinc-400" />}>
          Logger
        </NodeBadge>
        {lines.length > 0 && (
          <NodeBadge tone="info" accent="text-zinc-400">
            {lines.length}
          </NodeBadge>
        )}
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={280}
        minHeight={160}
        padded={false}
        subtleSelect
      >
        <div className="relative h-full overflow-hidden bg-black">
          {/* Strictly one entry per line - no wrapping, so a line never reflows into several as the
              node is resized. `nowheel` stops React Flow zooming instead of scrolling; `nodrag`
              keeps a drag-select inside the log from moving the node. */}
          <div
            ref={logRef}
            onScroll={(e) => {
              const el = e.currentTarget
              stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
            }}
            className="nowheel nodrag h-full cursor-text select-text overflow-auto px-2 pb-2 pt-2 font-mono text-[10px] leading-snug text-zinc-300"
          >
            {lines.length === 0 ? (
              <span className="text-zinc-500">
                {runId ? 'Waiting for output…' : 'Wire a node that streams a log'}
              </span>
            ) : (
              lines.slice(-SHOWN).map((line, i) => (
                <div key={i} className="-mx-1 whitespace-pre rounded-sm px-1 hover:bg-white/5">
                  {line}
                </div>
              ))
            )}
          </div>
          {lines.length > 0 && (
            <div className="absolute right-1.5 top-1.5 z-10 flex gap-1">
              <button
                onClick={() => void copy()}
                title="Copy the whole log"
                className="nodrag flex items-center gap-1 rounded border border-border bg-black/70 px-1.5 py-0.5 text-[10px] text-zinc-300 backdrop-blur hover:border-zinc-500 hover:text-white"
              >
                <CopyIcon className="h-3 w-3" />
                {copied ? 'Copied' : `Copy ${lines.length}`}
              </button>
              <button
                onClick={download}
                title="Save the whole log to a file"
                className="nodrag flex items-center rounded border border-border bg-black/70 px-1.5 py-0.5 text-zinc-300 backdrop-blur hover:border-zinc-500 hover:text-white"
              >
                <DownloadIcon className="h-3 w-3" />
              </button>
              <button
                onClick={() => runId && clear(runId)}
                title="Clear"
                className="nodrag flex items-center rounded border border-border bg-black/70 px-1.5 py-0.5 text-[10px] text-zinc-300 backdrop-blur hover:border-zinc-500 hover:text-white"
              >
                Clear
              </button>
            </div>
          )}
        </div>
      </NodeFrame>
      <PortHandle id="metrics" label="Log" kind="metrics" side="input" style={topStyle(0)} />
    </>
  )
}
