import { useEffect } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { isModelPort, portKindColor, type CorePort } from '@shared/coreNodes'
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import { useGenerationStore } from '../../../store/generationStore'
import { useGraphSelectionStore } from '../../../store/graphSelectionStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { activeDownload, useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { NodeFrame } from './NodeFrame'
import { NodeRunToolbar } from './NodeRunToolbar'
import {
  AdjustIcon,
  AlertIcon,
  BoxIcon,
  ImageGlyph,
  NodeBadge,
  NodeBadgeRow,
  SquareIcon,
  TypeIcon,
  WandIcon,
} from './NodeBadge'
import { resolveMedia } from '@/lib/media'

interface GraphNodeData extends Record<string, unknown> {
  itemId: string
}

// Handles are packed against an edge rather than spread down the whole side: content/signal ports
// stack from the top, model-family ports (model/vae/text-encoder) stack from the bottom — so model
// wiring reads as one band along the bottom and the image flow runs across the top.
const HANDLE_BASE = 18 // px from the packed edge to the first dot
const HANDLE_GAP = 22 // px between stacked dots

function topStyle(index: number): React.CSSProperties {
  return { top: HANDLE_BASE + index * HANDLE_GAP }
}

function bottomStyle(index: number): React.CSSProperties {
  return { top: 'auto', bottom: HANDLE_BASE + index * HANDLE_GAP }
}

/** One colored port dot with a hover chip naming the port — input (left) or output (right). */
function PortHandle({
  port,
  side,
  style,
}: {
  port: CorePort
  side: 'input' | 'output'
  style: React.CSSProperties
}): React.JSX.Element {
  const input = side === 'input'
  return (
    <Handle
      type={input ? 'target' : 'source'}
      id={port.id}
      position={input ? Position.Left : Position.Right}
      style={{ ...style, background: portKindColor(port.kind) }}
      className="group !h-3 !w-3 !border-2 !border-surface"
    >
      <span
        className={`pointer-events-none absolute top-1/2 z-50 hidden -translate-y-1/2 whitespace-nowrap rounded bg-black/90 px-1.5 py-0.5 text-[10px] leading-none text-zinc-100 shadow group-hover:block ${
          input ? 'right-full mr-2' : 'left-full ml-2'
        }`}
      >
        {port.label} <span className="text-zinc-400">· {port.kind}</span>
      </span>
    </Handle>
  )
}

/** Map a Core descriptor's `icon` string to a node-family glyph (falls back to the square). */
function coreGlyph(icon: string): React.JSX.Element {
  switch (icon) {
    case 'wand':
      return <WandIcon />
    case 'box':
      return <BoxIcon />
    case 'type':
      return <TypeIcon />
    case 'image':
      return <ImageGlyph />
    default:
      return <SquareIcon />
  }
}

/**
 * A generic Inline Core graph node backed by a `core` moodboard item. Resolves its descriptor from
 * the served `/v1/models` palette and renders in the same card style as the fal Generate node: a
 * floating title badge, an edge-to-edge output preview, and a footer with Run + an adjust (settings)
 * button. Params live behind the adjust button in the Core settings sidebar — the node face stays
 * clean, so a model node like Z-Image Turbo reads as one simple node. One colored handle per
 * declared port (inputs left, outputs right, colored by kind).
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
  const cancel = useGenerationStore((s) => s.cancel)
  const toggleSettings = useGenerationStore((s) => s.toggleCoreSettings)
  // This node is the selected graph's output node → it floats the graph's single Run control.
  const isRunTarget = useGraphSelectionStore((s) => s.runTargets.includes(itemId))
  const busy = useGenerationStore((s) => s.busyByFrame[itemId] ?? false)
  const progress = useGenerationStore((s) => s.progressByFrame[itemId])
  const status = useGenerationStore((s) => s.statusByFrame[itemId])

  // Model requirements (per node type) drive the blinking "missing models" hint + its popup, and
  // surface an in-node download indicator. Refetched when the model registry version changes (a
  // dropped-in file, or a completed download).
  const registryVersion = useCoreNodesStore((s) => s.registryVersion)
  const loadReqs = useModelRequirementsStore((s) => s.load)
  const openReqs = useModelRequirementsStore((s) => s.open)
  const reqs = useModelRequirementsStore((s) => (coreType ? s.byType[coreType] : undefined))
  const downloadsForType = useModelRequirementsStore((s) =>
    coreType ? s.downloads[coreType] : undefined,
  )
  useEffect(() => {
    if (coreType) void loadReqs(coreType)
  }, [coreType, registryVersion, loadReqs])

  if (!item || item.type !== 'core' || !item.data.core || !descriptor) {
    return (
      <NodeFrame id={id} selected={!!selected} minWidth={200} minHeight={92} subtleSelect>
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
  const pct = typeof progress === 'number' ? Math.round(progress * 100) : null

  // Take history for the on-node output strip (newest first). Older items predate history and only
  // carry a single `output` — treat that as a one-entry history. `output` marks the active take.
  const outputs = core.outputs ?? (core.output ? [core.output] : [])
  const activeTakeId = core.output?.takeId
  const setActiveOutput = (o: NonNullable<typeof core.output>): void => {
    void updateItem(itemId, { data: { ...item.data, core: { ...core, output: o } } })
  }

  // Real "models missing" signal from the requirements check (replaces the old options heuristic,
  // which wrongly assumed a silent auto-download). Assume OK until requirements load, to avoid a
  // hint flash on first render.
  const modelsMissing = reqs ? !reqs.allPresent : false
  const download = downloadsForType ? activeDownload(downloadsForType, reqs) : null
  const downloadPct = download ? Math.round(download.fraction * 100) : null

  // A loader/plumbing node (no media output) renders compact — no preview, no Run (it loads with
  // whatever downstream node runs). Generation nodes get the full preview card + the graph Run
  // control floated on the output node.
  const isLoader = descriptor.outputKind == null
  const fileParam = core?.params?.file
  const fileLabel = fileParam ? String(fileParam) : 'Auto'

  // Split each side into content (top-packed) and model-family (bottom-packed) ports.
  const inContent = descriptor.inputs.filter((p) => !isModelPort(p.kind))
  const inModel = descriptor.inputs.filter((p) => isModelPort(p.kind))
  const outContent = descriptor.outputs.filter((p) => !isModelPort(p.kind))
  const outModel = descriptor.outputs.filter((p) => isModelPort(p.kind))

  const handles = (
    <>
      {inContent.map((port, i) => (
        <PortHandle key={port.id} port={port} side="input" style={topStyle(i)} />
      ))}
      {inModel.map((port, i) => (
        <PortHandle key={port.id} port={port} side="input" style={bottomStyle(i)} />
      ))}
      {outContent.map((port, i) => (
        <PortHandle key={port.id} port={port} side="output" style={topStyle(i)} />
      ))}
      {outModel.map((port, i) => (
        <PortHandle key={port.id} port={port} side="output" style={bottomStyle(i)} />
      ))}
    </>
  )

  if (isLoader) {
    // A loader's whole job is picking a file, so its SELECT dropdown(s) live directly on the node
    // face (not behind Adjust) — the one exception to "params off the node face", which exists to
    // keep *generation* one-click. Any non-select params (rare for a loader) stay behind Adjust.
    const selectParams = descriptor.params.filter((p) => p.widget === 'select')
    const otherParams = descriptor.params.filter((p) => p.widget !== 'select')
    const setParam = (key: string, value: string): void => {
      void updateItem(itemId, {
        data: { ...item.data, core: { ...core, params: { ...core.params, [key]: value } } },
      })
    }
    return (
      <>
        <NodeBadgeRow>
          <NodeBadge icon={coreGlyph(descriptor.icon)}>{descriptor.title}</NodeBadge>
        </NodeBadgeRow>
        <NodeFrame
          id={id}
          selected={!!selected}
          minWidth={188}
          minHeight={44}
          padded={false}
          subtleSelect
        >
          <div className="flex h-full w-full flex-col gap-1 px-2 py-1.5">
            {selectParams.length > 0 ? (
              selectParams.map((field) => {
                const opts = field.options ?? []
                const hasAuto = opts.some((o) => o.value === '')
                return (
                  <select
                    key={field.key}
                    value={String(core.params?.[field.key] ?? field.default ?? '')}
                    onChange={(e) => setParam(field.key, e.target.value)}
                    title={field.label}
                    className="nodrag w-full min-w-0 rounded border border-border bg-panel px-1.5 py-1 text-[10px] text-zinc-200 outline-none focus:border-accent"
                  >
                    {/* Empty value = auto-pick the first file; shown as a "Select …" prompt. */}
                    {!hasAuto && <option value="">{`Select ${field.label}`}</option>}
                    {opts.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                )
              })
            ) : (
              <span
                className="min-w-0 flex-1 truncate px-1 font-mono text-[10px] text-zinc-400"
                title={fileLabel}
              >
                {fileLabel}
              </span>
            )}
            {otherParams.length > 0 && (
              <div className="flex justify-end">
                <button
                  onClick={() => toggleSettings(itemId)}
                  title="Settings"
                  data-gen-settings-toggle
                  className="nodrag flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
                >
                  <AdjustIcon />
                </button>
              </div>
            )}
          </div>
        </NodeFrame>
        {handles}
      </>
    )
  }

  return (
    <>
      {/* The graph's single Run control, floated above this output node while the graph is selected. */}
      <NodeRunToolbar
        isTarget={isRunTarget}
        busy={busy}
        onRun={() => void runWorkflow(itemId)}
        onStop={() => void cancel(itemId)}
        disabled={download !== null}
        disabledReason="Downloading model…"
      />
      {/* Floating title badge — matches the fal Generate node. */}
      <NodeBadgeRow>
        <NodeBadge icon={coreGlyph(descriptor.icon)}>{descriptor.title}</NodeBadge>
        {modelsMissing && (
          <button
            onClick={() => openReqs(descriptor.type)}
            title="Models missing — click to download"
            className="nodrag flex h-6 animate-pulse items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/10 px-2 text-[10px] font-medium text-amber-300 shadow-sm backdrop-blur hover:animate-none hover:bg-amber-500/20"
          >
            <AlertIcon className="h-3.5 w-3.5" />
            Models
          </button>
        )}
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={200}
        padded={false}
        subtleSelect
      >
        <div className="relative flex h-full w-full flex-col">
          {/* Edge-to-edge output preview. */}
          <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
            {core.output?.kind === 'image' ? (
              <img
                src={resolveMedia(core.output.filePath)}
                alt=""
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center px-4">
                <span className="text-center text-[10px] text-zinc-600">
                  {busy
                    ? (status ?? 'Working…')
                    : download
                      ? `Downloading ${download.label}…`
                      : modelsMissing
                        ? 'Models missing — click the hint to download'
                        : 'Run to generate'}
                </span>
              </div>
            )}

            {/* Busy = generating; else show a download indicator while a model is being fetched. */}
            {busy ? (
              <>
                <span className="absolute left-2 top-2 flex max-w-[calc(100%-1rem)] items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 text-[10px] font-medium text-emerald-300 backdrop-blur">
                  <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-emerald-400" />
                  <span className="truncate">
                    {status ?? (pct != null ? `${pct}%` : 'Working…')}
                  </span>
                </span>
                <div className="absolute inset-x-0 bottom-0 h-1 bg-black/40">
                  <div
                    className="h-full bg-emerald-400 transition-all"
                    style={{ width: `${pct ?? 0}%` }}
                  />
                </div>
              </>
            ) : (
              download && (
                <>
                  <span className="absolute left-2 top-2 flex max-w-[calc(100%-1rem)] items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 text-[10px] font-medium text-sky-300 backdrop-blur">
                    <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-sky-400" />
                    <span className="truncate">
                      {download.label} {downloadPct != null ? `${downloadPct}%` : ''}
                    </span>
                  </span>
                  <div className="absolute inset-x-0 bottom-0 h-1 bg-black/40">
                    <div
                      className="h-full bg-sky-400 transition-all"
                      style={{ width: `${downloadPct ?? 0}%` }}
                    />
                  </div>
                </>
              )
            )}
          </div>

          {/* Take history: every render this node produced, newest first. Click one to make it the
              active output (shown large + flowed downstream). Only shown once there's more than one. */}
          {outputs.length > 1 && (
            <div className="nowheel flex shrink-0 gap-1 overflow-x-auto border-t border-border bg-surface/90 px-1.5 py-1.5">
              {outputs.map((o) => (
                <button
                  key={o.takeId}
                  onClick={() => setActiveOutput(o)}
                  title={o.takeId === activeTakeId ? 'Active take' : 'Use this take'}
                  className={`nodrag relative h-11 w-11 shrink-0 overflow-hidden rounded border transition-colors ${
                    o.takeId === activeTakeId
                      ? 'border-emerald-400 ring-1 ring-emerald-400/40'
                      : 'border-border hover:border-zinc-500'
                  }`}
                >
                  {o.kind === 'image' ? (
                    <img
                      src={resolveMedia(o.filePath)}
                      alt=""
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center bg-black text-[9px] uppercase tracking-wide text-zinc-500">
                      {o.kind}
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}

          {/* Footer: category label + settings (adjust). Run lives on the graph's output node. */}
          <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border bg-surface/90 px-1.5 py-1">
            <span className="truncate px-1 text-[10px] uppercase tracking-wide text-zinc-500">
              {descriptor.category}
            </span>
            <button
              onClick={() => toggleSettings(itemId)}
              title="Settings"
              data-gen-settings-toggle
              className="nodrag flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
            >
              <AdjustIcon />
            </button>
          </div>
        </div>
      </NodeFrame>

      {handles}
    </>
  )
}
