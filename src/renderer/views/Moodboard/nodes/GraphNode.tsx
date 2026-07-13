import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { CoreParamField } from '@shared/coreNodes'
import { portKindColor } from '@shared/coreNodes'
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import { useGenerationStore } from '../../../store/generationStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { NodeFrame } from './NodeFrame'
import { resolveMedia } from '@/lib/media'

interface GraphNodeData extends Record<string, unknown> {
  itemId: string
}

/** Even vertical spacing for `count` handles down an edge. */
function edgePercent(index: number, count: number): string {
  return `${(((index + 1) / (count + 1)) * 100).toFixed(2)}%`
}

const WIDGET_CLASS =
  'nodrag w-full rounded border border-border bg-surface px-1.5 py-0.5 text-[11px] text-zinc-200'

function ParamWidget({
  field,
  value,
  onChange,
}: {
  field: CoreParamField
  value: unknown
  onChange: (v: string | number | boolean) => void
}): React.JSX.Element {
  if (field.widget === 'boolean') {
    return (
      <label className="flex items-center gap-1 text-[11px] text-zinc-300">
        <input
          type="checkbox"
          className="nodrag"
          checked={Boolean(value ?? field.default)}
          onChange={(e) => onChange(e.target.checked)}
        />
        {field.label}
      </label>
    )
  }
  if (field.widget === 'select') {
    return (
      <select
        className={WIDGET_CLASS}
        value={String(value ?? field.default)}
        onChange={(e) => onChange(e.target.value)}
      >
        {(field.options ?? []).map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    )
  }
  if (field.widget === 'number' || field.widget === 'seed') {
    return (
      <input
        type="number"
        className={WIDGET_CLASS}
        value={Number(value ?? field.default)}
        min={field.min}
        max={field.max}
        step={field.step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    )
  }
  if (field.widget === 'textarea') {
    return (
      <textarea
        className={`${WIDGET_CLASS} resize-none`}
        rows={2}
        value={String(value ?? field.default)}
        onChange={(e) => onChange(e.target.value)}
      />
    )
  }
  return (
    <input
      type="text"
      className={WIDGET_CLASS}
      value={String(value ?? field.default)}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}

/**
 * A generic Inline Core graph node backed by a `core` moodboard item. Resolves its descriptor from
 * the served `/v1/models` palette and renders one colored handle per declared port (inputs left,
 * outputs right, colored by kind) plus a param widget per non-advanced field. Param edits persist
 * to the item's `data.core.params`. This is the low-level workflow node (load/sample/encode/vae).
 */
export function GraphNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { itemId } = data as GraphNodeData
  const item = useMoodboardStore((s) => s.items.find((i) => i.id === itemId))
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const coreType = item?.type === 'core' ? item.data.core?.type : undefined
  const descriptor = useCoreNodesStore((s) =>
    coreType ? s.descriptors.find((d) => d.type === coreType) : undefined,
  )
  const runWorkflow = useGenerationStore((s) => s.runWorkflow)
  const busy = useGenerationStore((s) => s.busyByFrame[itemId] ?? false)

  if (!item || item.type !== 'core' || !item.data.core || !descriptor) {
    return (
      <NodeFrame id={id} selected={!!selected} minWidth={200} minHeight={92}>
        <div className="flex h-full flex-col items-center justify-center gap-1 p-3 text-center">
          {coreType ? (
            <>
              <span className="text-[11px] font-semibold text-amber-300">Node unavailable</span>
              <span className="text-[10px] leading-tight text-zinc-400">
                <span className="text-zinc-300">{coreType}</span> is not registered. Start Inline
                Core and install its runtime (the <span className="text-zinc-300">zimage</span>{' '}
                extra).
              </span>
            </>
          ) : (
            <span className="text-[11px] text-zinc-500">Core node</span>
          )}
        </div>
      </NodeFrame>
    )
  }

  const core = item.data.core
  const fields = descriptor.params.filter((p) => !p.advanced)
  const missingCategories = descriptor.params
    .filter((p) => p.optionsFrom && (p.options?.length ?? 0) === 0)
    .map((p) => p.optionsFrom as string)
  const setParam = (key: string, value: string | number | boolean): void => {
    void updateItem(itemId, {
      data: { ...item.data, core: { type: core.type, params: { ...core.params, [key]: value } } },
    })
  }

  return (
    <>
      <NodeFrame id={id} selected={!!selected} minWidth={180} minHeight={80}>
        <div className="flex h-full w-full flex-col gap-1.5 p-2">
          <div className="flex items-center justify-between gap-1">
            <span className="truncate text-[11px] font-semibold text-zinc-100">
              {descriptor.title}
            </span>
            <button
              onClick={() => void runWorkflow(itemId)}
              disabled={busy}
              title="Run up to this node"
              className="nodrag flex h-5 w-5 shrink-0 items-center justify-center rounded text-emerald-400 hover:bg-black/40 disabled:opacity-40"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="h-3.5 w-3.5">
                <path d="M8 5v14l11-7z" />
              </svg>
            </button>
          </div>
          {missingCategories.length > 0 && (
            <div className="nodrag rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-1">
              <span className="text-[10px] leading-tight text-amber-200">
                Model files not found ({missingCategories.join(', ')}). Add them to Inline
                Core&apos;s models folder, then reopen.
              </span>
            </div>
          )}
          {core.output?.kind === 'image' && (
            <img
              src={resolveMedia(core.output.filePath)}
              alt=""
              className="h-20 w-full rounded object-cover"
            />
          )}
          {fields.map((field) => (
            <div key={field.key} className="flex flex-col gap-0.5">
              {field.widget !== 'boolean' && (
                <span className="text-[9px] uppercase tracking-wide text-zinc-500">
                  {field.label}
                </span>
              )}
              <ParamWidget
                field={field}
                value={core.params[field.key]}
                onChange={(v) => setParam(field.key, v)}
              />
            </div>
          ))}
        </div>
      </NodeFrame>

      {descriptor.inputs.map((port, i) => (
        <Handle
          key={port.id}
          type="target"
          id={port.id}
          position={Position.Left}
          style={{
            top: edgePercent(i, descriptor.inputs.length),
            background: portKindColor(port.kind),
          }}
          className="group !h-3 !w-3 !border-2 !border-surface"
        >
          <span className="pointer-events-none absolute right-full top-1/2 z-50 mr-2 hidden -translate-y-1/2 whitespace-nowrap rounded bg-black/90 px-1.5 py-0.5 text-[10px] leading-none text-zinc-100 shadow group-hover:block">
            {port.label} <span className="text-zinc-400">· {port.kind}</span>
          </span>
        </Handle>
      ))}
      {descriptor.outputs.map((port, i) => (
        <Handle
          key={port.id}
          type="source"
          id={port.id}
          position={Position.Right}
          style={{
            top: edgePercent(i, descriptor.outputs.length),
            background: portKindColor(port.kind),
          }}
          className="group !h-3 !w-3 !border-2 !border-surface"
        >
          <span className="pointer-events-none absolute left-full top-1/2 z-50 ml-2 hidden -translate-y-1/2 whitespace-nowrap rounded bg-black/90 px-1.5 py-0.5 text-[10px] leading-none text-zinc-100 shadow group-hover:block">
            {port.label} <span className="text-zinc-400">· {port.kind}</span>
          </span>
        </Handle>
      ))}
    </>
  )
}
