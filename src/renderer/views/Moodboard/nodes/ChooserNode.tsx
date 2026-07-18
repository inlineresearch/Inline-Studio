import { useEffect, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { listNodeDefs, groupByOwner } from '@shared/nodes/registry'
import { useFrameStore } from '../../../store/frameStore'
import { useAssetStore } from '../../../store/assetStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import {
  getAssetDragIds,
  getFrameDragId,
  getOutputTakeId,
  ASSET_DND_TYPE,
  FRAME_DND_TYPE,
} from '../../../lib/dnd'
import { NodeFrame } from './NodeFrame'
import { ChevronDownIcon, FilmIcon, LinkIcon, NodeBadge, NodeBadgeRow } from './NodeBadge'
import { ThumbStrip } from './ThumbStrip'
import { resolveInputThumbs } from './inputThumbs'

interface ChooserNodeData extends Record<string, unknown> {
  frameId: string
}

/** Two-sparkle mark flagging the fal (AI/API) path. Matches the Generate node + toolbar button. */
function SparkleIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-4 w-4 shrink-0"
    >
      <path d="M12 3l1.9 4.8L18.6 9.7 13.9 11.6 12 16.4 10.1 11.6 5.4 9.7 10.1 7.8Z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z" />
    </svg>
  )
}

/**
 * A freshly-created, engine-unset frame (`provider:'unset'`): the single unified generation node
 * before the user picks how it renders. It offers two paths - ComfyUI Workflow (becomes a FrameNode)
 * or Start with Fal API, which opens a model dropdown under the button; picking a model turns it into
 * a GenNode on that model. Inputs already dropped/wired show at the top and carry over either way.
 */
export function ChooserNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { frameId } = data as ChooserNodeData
  const frame = useFrameStore((s) => s.frames.find((f) => f.id === frameId))
  const inputs = useFrameStore((s) => s.inputsByFrame[frameId]) ?? []
  const setProvider = useFrameStore((s) => s.setProvider)
  const addInputs = useFrameStore((s) => s.addInputs)
  const addSourceInput = useFrameStore((s) => s.addSourceInput)
  const removeInputById = useFrameStore((s) => s.removeInputById)
  const setHero = useFrameStore((s) => s.setHero)
  const allFrames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)
  const inputsByFrame = useFrameStore((s) => s.inputsByFrame)
  const assets = useAssetStore((s) => s.assets)
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const [dropActive, setDropActive] = useState(false)
  // Whether the fal model list (a dropdown under the "Start with Fal API" button) is open.
  const [modelsOpen, setModelsOpen] = useState(false)
  // The button + dropdown wrapper, so a click anywhere outside it closes the list.
  const falMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!modelsOpen) return
    const onPointerDown = (e: PointerEvent): void => {
      if (!falMenuRef.current?.contains(e.target as Node)) setModelsOpen(false)
    }
    // Capture phase so we still catch the click even if the canvas stops its propagation.
    document.addEventListener('pointerdown', onPointerDown, true)
    return () => document.removeEventListener('pointerdown', onPointerDown, true)
  }, [modelsOpen])

  const inputThumbs = resolveInputThumbs(inputs, { assets, allFrames, takesByFrame, inputsByFrame })

  // Choose ComfyUI: just switch the node into a comfy frame. It renders the FrameNode with its own
  // "Link Workflow" badge - the user clicks that to link + open the Generate tab. We don't
  // navigate automatically, so picking the engine and linking a workflow stay separate steps.
  const onChooseComfy = (): void => {
    void setProvider(frameId, 'comfy')
  }

  // Pick a fal model: the node becomes a GenNode on that model (params/output-kind seeded from the
  // def) and reshapes to the Generate node's portrait proportions (output preview is the hero).
  const onPickFalModel = (modelId: string): void => {
    void setProvider(frameId, 'fal', modelId)
    void updateItem(id, { width: 240, height: 380 }, false)
  }

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
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<FilmIcon />} title={frame ? `Frame ${frame.name}` : undefined}>
          Frame {frame?.name ?? '—'}
        </NodeBadge>
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={170}
        padded={false}
        subtleSelect
        overflowVisible={modelsOpen}
      >
        <div
          className="relative flex h-full w-full flex-col"
          onDragOver={onDragOver}
          onDragLeave={() => setDropActive(false)}
          onDrop={onDrop}
        >
          <div
            className={`relative flex min-h-0 flex-1 flex-col bg-surface/60 ${modelsOpen ? 'overflow-visible' : 'overflow-hidden'}`}
          >
            <div className="relative flex flex-1 flex-col items-center justify-center gap-2.5 p-3">
              {/* Inputs already fed in - shown at the top, removable, carried into either engine. */}
              <ThumbStrip
                items={inputThumbs.map((t) => ({
                  id: t.id,
                  url: t.url,
                  kind: t.kind,
                  poster: t.poster,
                }))}
                onRemove={(i) => void removeInputById(frameId, inputThumbs[i].id)}
                edge="top"
              />

              <span className="text-[11px] font-medium text-zinc-400">Choose how to generate</span>
              <div className="flex w-full max-w-[220px] flex-col gap-2">
                <button
                  onClick={onChooseComfy}
                  className="nodrag flex items-center justify-center gap-2 rounded-md border border-blue-500/40 bg-blue-500/10 px-3 py-2 text-[12px] font-medium text-blue-200 hover:bg-blue-500/20"
                >
                  <LinkIcon className="h-3.5 w-3.5" />
                  ComfyUI Workflow
                </button>

                {/* "Start with Fal API" + its model dropdown, anchored right under the button, in the
                    button's emerald theme + width; it's free to spill past the frame (overflowVisible). */}
                <div ref={falMenuRef} className="relative w-full">
                  <button
                    onClick={() => setModelsOpen((v) => !v)}
                    className={`nodrag flex w-full items-center justify-center gap-2 border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-[12px] font-medium text-emerald-200 hover:bg-emerald-500/20 ${modelsOpen ? 'rounded-t-md border-b-transparent' : 'rounded-md'}`}
                  >
                    <SparkleIcon />
                    Start with Fal API
                    <ChevronDownIcon
                      className={`h-3 w-3 transition-transform ${modelsOpen ? 'rotate-180' : ''}`}
                    />
                  </button>
                  {modelsOpen && (
                    <div className="nodrag nowheel absolute left-0 right-0 top-full z-30 max-h-56 overflow-y-auto rounded-b-md border border-t-0 border-emerald-500/40 bg-panel py-0.5 shadow-xl shadow-black/40">
                      {groupByOwner(listNodeDefs()).map((group) => (
                        <div key={group.owner}>
                          <div className="px-2.5 pb-0.5 pt-1.5 text-[9px] font-semibold uppercase tracking-wide text-emerald-300/70">
                            {group.label}
                          </div>
                          {group.defs.map((d) => (
                            <button
                              key={d.id}
                              onClick={() => onPickFalModel(d.id)}
                              className="flex w-full flex-col items-start px-2.5 py-1.5 text-left hover:bg-emerald-500/10"
                            >
                              <span className="w-full truncate text-[11px] text-zinc-100">
                                {d.title}
                              </span>
                              <span className="text-[10px] text-zinc-500">{d.category}</span>
                            </button>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {dropActive && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-md border-2 border-dashed border-accent bg-accent/15 text-[11px] font-medium text-panel">
              Add as input
            </div>
          )}
        </div>
      </NodeFrame>

      {/* Same Input/Output handles as a frame, so wiring works before an engine is chosen. */}
      <Handle
        type="target"
        id="in"
        position={Position.Left}
        title="Input"
        className="group !h-3 !w-3 !border-2 !border-surface !bg-emerald-400"
      >
        <span className="pointer-events-none absolute right-full top-1/2 mr-1.5 hidden -translate-y-1/2 items-center whitespace-nowrap rounded-md border border-border bg-panel/95 px-1.5 py-0.5 text-[10px] font-medium text-zinc-200 shadow-sm backdrop-blur group-hover:flex">
          Input
        </span>
      </Handle>
      <Handle
        type="source"
        id="out"
        position={Position.Right}
        title="Output"
        className="group !h-3 !w-3 !border-2 !border-surface !bg-indigo-400"
      >
        <span className="pointer-events-none absolute left-full top-1/2 ml-1.5 hidden -translate-y-1/2 items-center whitespace-nowrap rounded-md border border-border bg-panel/95 px-1.5 py-0.5 text-[10px] font-medium text-zinc-200 shadow-sm backdrop-blur group-hover:flex">
          Output
        </span>
      </Handle>
    </>
  )
}
