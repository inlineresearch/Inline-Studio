import { useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { takeWaveformPath } from '@shared/media'
import { getNodeDef } from '@shared/nodes/registry'
import { dotPorts, handleIdForPort, portIdForHandle } from '@shared/nodes/handles'
import { formatPrice, mediaFamily } from '@shared/nodes/types'
import { useFrameStore } from '../../../store/frameStore'
import { useAssetStore } from '../../../store/assetStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useGenerationStore } from '../../../store/generationStore'
import { useGraphSelectionStore } from '../../../store/graphSelectionStore'
import { useLightboxStore } from '../../../store/lightboxStore'
import {
  getAssetDragIds,
  getFrameDragId,
  getOutputTakeId,
  ASSET_DND_TYPE,
  FRAME_DND_TYPE,
} from '../../../lib/dnd'
import { useMediaContextMenu } from '../../../lib/mediaContextMenu'
import { VideoPreview } from '../../../components/VideoPreview'
import { Waveform } from '../../../components/Waveform'
import { NodeFrame } from './NodeFrame'
import {
  AdjustIcon,
  AudioGlyph,
  ImageGlyph,
  NodeBadge,
  NodeBadgeRow,
  VideoGlyph,
} from './NodeBadge'
import { ThumbStrip } from './ThumbStrip'
import { NodeRunToolbar } from './NodeRunToolbar'
import { useGraphMenu } from './useGraphMenu'
import { resolveInputThumbs } from './inputThumbs'
import { ModelPicker } from '../ModelPicker'
import { resolveMedia } from '@/lib/media'

interface GenNodeData extends Record<string, unknown> {
  frameId: string
}

/** Two-sparkle mark flagging an AI/API-backed node. Matches the toolbar's create button. */
function SparkleIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3.5 w-3.5 shrink-0 text-emerald-400"
    >
      <path d="M12 3l1.9 4.8L18.6 9.7 13.9 11.6 12 16.4 10.1 11.6 5.4 9.7 10.1 7.8Z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z" />
    </svg>
  )
}

/**
 * A left-edge input dot that reveals a small hint chip on hover - an icon + label naming what to
 * connect (a "T" for the text prompt, an image/video icon for the media input).
 */
function InputHandle({
  id,
  top,
  colorClass,
  icon,
  label,
}: {
  id: string
  top: string
  colorClass: string
  icon: React.ReactNode
  label: string
}): React.JSX.Element {
  return (
    <Handle
      type="target"
      id={id}
      position={Position.Left}
      style={{ top }}
      title={label}
      className={`group !h-3 !w-3 !border-2 !border-surface ${colorClass}`}
    >
      <span className="pointer-events-none absolute right-full top-1/2 mr-1.5 hidden -translate-y-1/2 items-center gap-1 whitespace-nowrap rounded-md border border-border bg-panel/95 px-1.5 py-0.5 text-[10px] font-medium text-zinc-200 shadow-sm backdrop-blur group-hover:flex">
        {icon}
        {label}
      </span>
    </Handle>
  )
}

/**
 * A "Generate" node: a portrait card whose body is the edge-to-edge output preview, a title badge
 * + live price estimate above it, a selection-only play control on the right, and a footer that
 * holds the model picker + a settings (adjust) button. Inputs are bare dots - a prompt (fed by a
 * Prompt node) and, for image models, an image. Backed by a `provider:'fal'` frame.
 */
export function GenNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { frameId } = data as GenNodeData
  const frame = useFrameStore((s) => s.frames.find((f) => f.id === frameId))
  const takes = useFrameStore((s) => s.takesByFrame[frameId]) ?? []
  const inputs = useFrameStore((s) => s.inputsByFrame[frameId]) ?? []
  const addInputs = useFrameStore((s) => s.addInputs)
  const addSourceInput = useFrameStore((s) => s.addSourceInput)
  const removeInputById = useFrameStore((s) => s.removeInputById)
  const setHero = useFrameStore((s) => s.setHero)
  const allFrames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)
  const inputsByFrame = useFrameStore((s) => s.inputsByFrame)
  const assets = useAssetStore((s) => s.assets)
  const connectors = useMoodboardStore((s) => s.connectors)
  const run = useGenerationStore((s) => s.run)
  const cancel = useGenerationStore((s) => s.cancel)
  const setModel = useGenerationStore((s) => s.setModel)
  const toggleSettings = useGenerationStore((s) => s.toggleSettings)
  const openModelInfo = useGenerationStore((s) => s.openModelInfo)
  const busy = useGenerationStore((s) => s.busyByFrame[frameId] ?? false)
  const progress = useGenerationStore((s) => s.progressByFrame[frameId])
  const status = useGenerationStore((s) => s.statusByFrame[frameId])
  // This node is the selected graph's output node → it floats the graph's single Run control.
  const isRunTarget = useGraphSelectionStore((s) => s.runTargets.includes(id))
  // Hooks cannot be conditional, so the hero take is resolved before the early return below.
  const heroTake = takes.find((t) => t.id === frame?.heroTakeId) ?? takes[0]
  const graphMenu = useGraphMenu(id, 'graph', heroTake?.kind === 'image' ? heroTake.id : undefined)
  const onMediaContextMenu = useMediaContextMenu()
  const openLightbox = useLightboxStore((s) => s.open)
  const [dropActive, setDropActive] = useState(false)
  // Which take is shown as the large preview (index into the hero-first ordering below).
  const [takeIdx, setTakeIdx] = useState(0)

  const def = frame?.modelId ? getNodeDef(frame.modelId) : undefined

  if (!frame || !def) {
    return (
      <NodeFrame id={id} selected={!!selected} minWidth={200} minHeight={120} subtleSelect>
        <div className="flex h-full items-center justify-center p-3 text-center text-[11px] text-zinc-500">
          Unknown generation model.
        </div>
      </NodeFrame>
    )
  }

  // Takes newest-first with the hero floated to the front, so the chosen output shows by default.
  const ordered = [...takes]
  if (frame.heroTakeId) {
    const i = ordered.findIndex((t) => t.id === frame.heroTakeId)
    if (i > 0) ordered.unshift(ordered.splice(i, 1)[0])
  }
  const safeTakeIdx = ordered.length ? Math.min(takeIdx, ordered.length - 1) : 0
  const take = ordered.length ? ordered[safeTakeIdx] : undefined
  // The node's wired/dropped inputs, resolved to thumbnails (same as the Frame node). Shown as a
  // strip at the top so dropping several images one-by-one visibly accumulates and each is removable.
  const inputThumbs = resolveInputThumbs(inputs, { assets, allFrames, takesByFrame, inputsByFrame })
  // Caption a thumb with its port only when the model has more than one dot to tell apart.
  const portLabels = new Map(
    def.inputs.length > 1
      ? inputs.flatMap((i) => {
          const port = def.inputs.find((p) => p.id === i.handle)
          return port ? [[i.id, port.label] as const] : []
        })
      : [],
  )
  // Input dots are driven by the model's API: one dot per declared input port, so a model with two
  // image inputs (a start and an end keyframe) gets two dots the user can tell apart.
  const ports = dotPorts(def)
  const hasPrompt = connectors.some(
    (c) => c.toItemId === id && (c.data?.targetHandle as string | undefined) === 'prompt',
  )
  const wiredHandles = new Set(
    connectors
      .filter((c) => c.toItemId === id)
      .map((c) => portIdForHandle(def, c.data?.targetHandle as string | undefined))
      .filter((p): p is string => !!p),
  )
  // A required port counts as satisfied by anything untagged too: dropped inputs carry no port.
  const untagged = inputs.some((i) => !i.handle)
  const unmet = ports.filter(
    (p) => p.required && !wiredHandles.has(p.id) && !inputs.some((i) => i.handle === p.id),
  )
  const missing =
    !hasPrompt && !def.promptOptional
      ? 'Connect a Prompt node'
      : unmet.length > 0 && !untagged
        ? `Wire ${unmet.map((p) => p.label.toLowerCase()).join(' & ')}`
        : null
  const pct = typeof progress === 'number' ? Math.round(progress * 100) : null
  const price = def.estimatePrice?.(frame.params) ?? null

  // Left-edge input dots, in order - each only appears when the model's API takes that input.
  // They're spaced evenly down the edge (rendered below), so 1/2/3 dots lay out cleanly.
  const inputDots = [
    {
      id: 'prompt',
      colorClass: '!bg-amber-400',
      icon: <span className="text-xs font-bold leading-none">T</span>,
      label: def.promptOptional ? 'Prompt (optional)' : 'Prompt',
    },
    ...ports.map((p) => {
      const family = mediaFamily(p.kind)
      return {
        id: handleIdForPort(def, p),
        colorClass: family === 'audio' ? '!bg-violet-400' : '!bg-emerald-400',
        icon:
          family === 'audio' ? (
            <AudioGlyph className="h-3.5 w-3.5" />
          ) : family === 'video' ? (
            <VideoGlyph className="h-3.5 w-3.5" />
          ) : (
            <ImageGlyph className="h-3.5 w-3.5" />
          ),
        label: p.required ? p.label : `${p.label} (optional)`,
      }
    }),
  ]

  const canDrop = (e: React.DragEvent): boolean =>
    e.dataTransfer.types.includes(ASSET_DND_TYPE) || e.dataTransfer.types.includes(FRAME_DND_TYPE)
  const onDragOver = (e: React.DragEvent): void => {
    if (!canDrop(e)) return
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    if (!dropActive) setDropActive(true)
  }
  const onDrop = (e: React.DragEvent): void => {
    if (!canDrop(e)) return
    e.preventDefault()
    e.stopPropagation()
    setDropActive(false)
    const droppedFrameId = getFrameDragId(e.dataTransfer)
    if (droppedFrameId) {
      if (droppedFrameId !== frameId) {
        // A specific output take dragged in → pin it as the source's hero so this input uses it.
        const takeId = getOutputTakeId(e.dataTransfer)
        if (takeId) void setHero(droppedFrameId, takeId)
        void addSourceInput(frameId, droppedFrameId)
      }
      return
    }
    const existing = new Set(inputs.map((i) => i.assetId))
    const ids = getAssetDragIds(e.dataTransfer).filter((x) => !existing.has(x))
    if (ids.length) void addInputs(frameId, ids)
  }

  return (
    <>
      {/* The graph's single Run control, floated above this output node while the graph is selected. */}
      <NodeRunToolbar
        isTarget={isRunTarget}
        busy={busy}
        onRun={() => void run(frameId)}
        onStop={() => void cancel(frameId)}
        disabled={!!missing}
        disabledReason={missing ?? undefined}
        menuItems={graphMenu.items}
        menuNote={graphMenu.note}
      />
      {/* Title + live price-estimate badges - float above the node. */}
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<SparkleIcon />}>Generate</NodeBadge>
        {price && (
          <NodeBadge
            tone="info"
            accent="text-emerald-300"
            tooltip="Rough estimate only - the actual generation cost can differ with resolution, output size and options. Check fal.ai's pricing page for accurate, up-to-date rates."
          >
            {formatPrice(price)}
          </NodeBadge>
        )}
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={220}
        padded={false}
        subtleSelect
      >
        <div
          className="relative flex h-full w-full flex-col"
          onDragOver={onDragOver}
          onDragLeave={() => setDropActive(false)}
          onDrop={onDrop}
        >
          {/* Edge-to-edge output preview. */}
          <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
            {/* Input thumbnails (top) - dropped/wired inputs. Click one to open it big in the
                zoomable lightbox (the strip is too small to read); hover an item to remove it. */}
            <ThumbStrip
              items={inputThumbs.map((t) => ({
                id: t.id,
                url: t.url,
                kind: t.kind,
                poster: t.poster,
                label: portLabels.get(t.id),
              }))}
              onSelect={(i) => {
                const t = inputThumbs[i]
                // The lightbox shows image/video; audio inputs have nothing to zoom into.
                if (t.kind === 'audio') return
                openLightbox({ src: t.saveSrc, kind: t.kind, name: `${def.title} input` })
              }}
              onRemove={(i) => void removeInputById(frameId, inputThumbs[i].id)}
              edge="top"
            />
            {take ? (
              take.kind === 'video' ? (
                <VideoPreview
                  src={resolveMedia(take.filePath)}
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: resolveMedia(take.filePath),
                      name: def.title,
                      kind: 'video',
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: resolveMedia(take.filePath),
                      kind: 'video',
                      name: def.title,
                    })
                  }
                  className="h-full w-full object-cover"
                />
              ) : take.kind === 'audio' ? (
                <div className="flex h-full w-full items-center px-3">
                  <Waveform
                    url={resolveMedia(takeWaveformPath(take.id))}
                    className="h-1/2 w-full text-emerald-400"
                  />
                </div>
              ) : (
                <img
                  src={resolveMedia(take.filePath)}
                  alt=""
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: resolveMedia(take.filePath),
                      name: def.title,
                      kind: 'image',
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: resolveMedia(take.filePath),
                      kind: 'image',
                      name: def.title,
                    })
                  }
                  className="h-full w-full object-cover"
                />
              )
            ) : (
              <div className="flex h-full w-full items-center justify-center px-4">
                <span className="text-center text-[10px] text-zinc-600">
                  {busy ? (status ?? 'Generating…') : (missing ?? 'Run to generate')}
                </span>
              </div>
            )}

            {busy && (
              <>
                {/* Live phase + progress (e.g. "Queued", "Generating 40%") - always visible, even
                    over a previous take while re-generating. */}
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
            )}

            {/* Multiple takes → a thumbnail strip; click one to make it this node's chosen output
                (its hero), so the shown image is what flows to anything wired downstream. */}
            {!busy && (
              <ThumbStrip
                items={ordered.map((t) => ({
                  id: t.id,
                  url: resolveMedia(t.filePath),
                  kind: t.kind,
                }))}
                selected={safeTakeIdx}
                onSelect={(i) => {
                  void setHero(frameId, ordered[i].id)
                  setTakeIdx(0)
                }}
              />
            )}
          </div>

          {/* Footer: model picker + settings (adjust). Run lives on the graph's output node. */}
          <div className="flex shrink-0 items-center justify-between gap-2 border-t border-border bg-surface/90 px-1.5 py-1">
            <ModelPicker
              modelId={frame.modelId}
              onSelect={(m) => void setModel(frameId, m)}
              onShowInfo={openModelInfo}
            />
            <button
              onClick={() => toggleSettings(frameId)}
              title="Settings"
              data-gen-settings-toggle
              className="nodrag flex h-6 w-6 shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-black/40 hover:text-zinc-100"
            >
              <AdjustIcon />
            </button>
          </div>

          {dropActive && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center border-2 border-dashed border-accent bg-accent/15 text-[11px] font-medium text-panel">
              Add as input
            </div>
          )}
        </div>
      </NodeFrame>

      {/* Input dots - hover reveals what to connect. Prompt (amber, text) is universal; media
          (emerald, image/video) and audio (violet) appear only when the model declares that input.
          Evenly spaced down the left edge. Output: indigo. */}
      {inputDots.map((d, i) => (
        <InputHandle
          key={d.id}
          id={d.id}
          top={`${18 + i * 22}px`}
          colorClass={d.colorClass}
          icon={d.icon}
          label={d.label}
        />
      ))}
      <Handle
        type="source"
        id="out"
        position={Position.Right}
        className="!h-3 !w-3 !border-2 !border-surface !bg-indigo-400"
      />
    </>
  )
}
