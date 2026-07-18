import { useLayoutEffect, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useFrameStore } from '../../../store/frameStore'
import { useAssetStore } from '../../../store/assetStore'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useUiStore } from '../../../store/uiStore'
import {
  getAssetDragIds,
  getFrameDragId,
  getMediaFileDrag,
  getOutputTakeId,
  ASSET_DND_TYPE,
  FRAME_DND_TYPE,
  MEDIA_FILE_DND_TYPE,
} from '../../../lib/dnd'
import { useMediaContextMenu } from '../../../lib/mediaContextMenu'
import { resolveMedia } from '@/lib/media'
import {
  importFilesToLibrary,
  importMediaUrlToLibrary,
  pickFilesViaInput,
} from '../../../lib/importFiles'
import { useLightboxStore } from '../../../store/lightboxStore'
import { requireComfyConnected } from '../../../lib/requireComfyConnected'
import { VideoPreview } from '../../../components/VideoPreview'
import { Waveform } from '../../../components/Waveform'
import { NodeFrame } from './NodeFrame'
import { FilmIcon, LinkIcon, NodeBadge, NodeBadgeRow, StarIcon, UploadIcon } from './NodeBadge'
import { ThumbStrip } from './ThumbStrip'
import { resolveInputThumbs } from './inputThumbs'

interface FrameNodeData extends Record<string, unknown> {
  frameId: string
}

// Bounds for the media body when fitting to a media's aspect ratio - keeps very
// wide/tall inputs from collapsing or ballooning the node.
const MIN_BODY = 160
const MAX_BODY = 480

/**
 * A frame on the canvas, styled like a preview: the body shows the frame's hero
 * input (carousel + "set as hero" when it has several). The header carries the
 * functional Output handle (wire it to a Preview/output node to see the result).
 * Three side handles allow purely-visual frame↔frame links (Miro-style).
 */
export function FrameNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { frameId } = data as FrameNodeData
  const frame = useFrameStore((s) => s.frames.find((sh) => sh.id === frameId))
  const inputs = useFrameStore((s) => s.inputsByFrame[frameId]) ?? []
  const busy = useFrameStore((s) => s.busyId === frameId)
  const linkFrame = useFrameStore((s) => s.linkFrame)
  const uploadInputs = useFrameStore((s) => s.uploadInputs)
  const reorderInputs = useFrameStore((s) => s.reorderInputs)
  const addInputs = useFrameStore((s) => s.addInputs)
  const addSourceInput = useFrameStore((s) => s.addSourceInput)
  const removeInputById = useFrameStore((s) => s.removeInputById)
  const setHero = useFrameStore((s) => s.setHero)
  const allFrames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)
  const inputsByFrame = useFrameStore((s) => s.inputsByFrame)
  const assets = useAssetStore((s) => s.assets)
  const item = useMoodboardStore((s) => s.items.find((it) => it.id === id))
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const setMode = useUiStore((s) => s.setMode)
  const setLinkedWorkflow = useUiStore((s) => s.setLinkedWorkflow)
  const setActiveFrame = useUiStore((s) => s.setActiveFrame)
  const onMediaContextMenu = useMediaContextMenu()
  const openLightbox = useLightboxStore((s) => s.open)
  const [idx, setIdx] = useState(0)
  // True while assets are dragged over the frame - highlights it as a drop target.
  const [dropActive, setDropActive] = useState(false)
  // Aspect ratio of the current media; drives the node height so the image fills
  // the body with no black letterboxing.
  const [aspect, setAspect] = useState<number | null>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  // Signature of the last applied fit (aspect + width); guards against re-firing
  // the height update on every render, which would loop and freeze the canvas.
  const lastFit = useRef<string>('')

  // Resolve each input to a thumbnail (asset media, or a flow link's hero take) - shared with
  // the Generate node so both surface their inputs identically.
  const thumbs = resolveInputThumbs(inputs, { assets, allFrames, takesByFrame, inputsByFrame })
  const count = thumbs.length
  const safeIdx = count ? Math.min(idx, count - 1) : 0
  const cur = count ? thumbs[safeIdx] : undefined
  const linked = !!frame?.comfyWorkflowName
  // A "Load Assets" loader: a pure viewer with no generation. It's freely resizable (the aspect
  // auto-fit below is skipped) and passes its loaded asset straight through as its output.
  const isLoader = !!item?.data.loader
  const mediaFit = isLoader ? 'object-contain' : 'object-cover'

  // Fit the node height to the media's aspect ratio at the current width, so the
  // body shows the image edge-to-edge with no black bars. The `lastFit` guard makes
  // this fire at most once per (aspect, width) pair - so the resulting height change
  // (which re-renders this node) can never feed back into another resize.
  const itemWidth = item?.width
  const itemHeight = item?.height
  useLayoutEffect(() => {
    const body = bodyRef.current
    // Loaders are freely sized by the user - never snap their height back to the media aspect.
    if (isLoader) return
    if (!aspect || !body || itemHeight == null || itemWidth == null) return
    const sig = `${aspect.toFixed(4)}:${itemWidth}`
    if (lastFit.current === sig) return
    const width = body.clientWidth
    if (!width) return
    lastFit.current = sig
    const targetBody = Math.max(MIN_BODY, Math.min(MAX_BODY, width / aspect))
    const delta = targetBody - body.clientHeight
    if (Math.abs(delta) < 1) return
    // Programmatic layout fit - don't pollute the undo history.
    void updateItem(id, { height: Math.round(itemHeight + delta) }, false)
  }, [aspect, itemWidth, itemHeight, id, updateItem, isLoader])

  // Drop the aspect lock when the visible input isn't an image/video (audio or none).
  const curKind = cur?.kind
  useLayoutEffect(() => {
    if (curKind !== 'image' && curKind !== 'video') setAspect(null)
  }, [curKind])

  const onLink = async (): Promise<void> => {
    if (!frame) return
    // ComfyUI must be reachable to link/open a workflow - otherwise send the user to the
    // Generate tab to connect first.
    if (!(await requireComfyConnected(() => setMode('generate')))) return
    const result = await linkFrame(frame.id)
    setLinkedWorkflow(result?.comfyWorkflowName ?? frame.comfyWorkflowName)
    setActiveFrame(frame.id)
    setMode('generate')
    // Push this frame's inputs to ComfyUI so they're available in LoadImage - the
    // cloud-safe path (no shared local folder needed). Best-effort.
    void uploadInputs(frame.id)
  }

  // Move the current input to the front. Reordering is keyed by asset id, so it only
  // applies when every input is asset-backed (flow inputs have no asset id).
  const canReorder = thumbs.every((t) => t.assetId)
  const makeHero = (): void => {
    if (!cur || safeIdx === 0 || !canReorder) return
    const ordered = thumbs
      .slice(safeIdx, safeIdx + 1)
      .concat(thumbs.filter((_, i) => i !== safeIdx))
      .map((t) => t.assetId as string)
    void reorderInputs(frameId, ordered)
    setIdx(0)
  }

  // Import files chosen from the OS (drag-drop or the "Select from Local" button) into the library,
  // then wire them as this frame's inputs.
  const addLocalFiles = async (files: File[]): Promise<void> => {
    if (files.length === 0) return
    const added = await importFilesToLibrary(files, null)
    if (added.length) {
      await useAssetStore.getState().load() // surface the new assets in the Library panel too
      void addInputs(
        frameId,
        added.map((a) => a.id),
      )
    }
  }
  const selectLocal = (): void => {
    void pickFilesViaInput().then(addLocalFiles)
  }

  // Accept Library assets, another node's output (a frame/Output tile), AND OS files (Finder/
  // Explorer) dropped onto the frame as inputs - dropping accumulates. stopPropagation keeps the
  // canvas from also handling the drop (which would spawn new frames). Already-present ones skip.
  const isFileDrag = (e: React.DragEvent): boolean => e.dataTransfer.types.includes('Files')
  const canDrop = (e: React.DragEvent): boolean =>
    e.dataTransfer.types.includes(ASSET_DND_TYPE) ||
    e.dataTransfer.types.includes(FRAME_DND_TYPE) ||
    e.dataTransfer.types.includes(MEDIA_FILE_DND_TYPE) ||
    isFileDrag(e)

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
    // OS files (Finder/Explorer) → import into the library, then add as inputs.
    if (isFileDrag(e)) {
      void addLocalFiles(Array.from(e.dataTransfer.files ?? []))
      return
    }
    // A Core-node output (raw media file) → import it into the library, then add as an input.
    const media = getMediaFileDrag(e.dataTransfer)
    if (media) {
      void (async () => {
        const asset = await importMediaUrlToLibrary(resolveMedia(media.filePath), media.name)
        if (asset) void addInputs(frameId, [asset.id])
      })()
      return
    }
    // A frame/Output tile → wire it as a flow-link input (resolves to its hero take). If a specific
    // output take was dragged, pin it as the source's hero so this input shows that exact image.
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
    const ids = getAssetDragIds(e.dataTransfer).filter((id) => !existing.has(id))
    if (ids.length) void addInputs(frameId, ids)
  }

  return (
    <>
      {/* Title + workflow-link badges - float above the node, matching the Generate node. A loader
          is a pure viewer, so it drops the generation (workflow-link) control. */}
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge
          icon={<FilmIcon />}
          title={isLoader ? 'Load Assets' : frame ? `Frame ${frame.name}` : undefined}
        >
          {isLoader ? 'Load Assets' : `Frame ${frame?.name ?? '—'}`}
        </NodeBadge>
        {!isLoader && (
          <button
            onClick={() => void onLink()}
            disabled={busy}
            title={linked ? 'Open the linked ComfyUI workflow' : 'Link a ComfyUI workflow'}
            className="nodrag flex h-6 items-center gap-1 rounded-full border border-blue-500/40 bg-blue-500/10 px-2 text-[10px] font-medium text-blue-300 shadow-sm backdrop-blur hover:bg-blue-500/20 disabled:opacity-40"
          >
            {busy ? (
              '…'
            ) : (
              <>
                <LinkIcon className="h-3 w-3" />
                {linked ? 'Open Workflow' : 'Link Workflow'}
              </>
            )}
          </button>
        )}
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={200}
        minHeight={170}
        padded={false}
        subtleSelect
      >
        <div
          className="relative flex h-full w-full flex-col"
          onDragOver={onDragOver}
          onDragLeave={() => setDropActive(false)}
          onDrop={onDrop}
        >
          <div
            ref={bodyRef}
            className="relative flex flex-1 items-center justify-center overflow-hidden bg-black"
          >
            {cur ? (
              cur.kind === 'video' ? (
                // `cur.url` is the playable source (transcoded preview when needed);
                // the poster shows while that transcode is still in progress.
                <VideoPreview
                  src={cur.url}
                  poster={cur.poster}
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget
                    if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight)
                  }}
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: cur.saveSrc,
                      name: frame ? `Frame ${frame.name}` : 'input',
                      kind: 'video',
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: cur.saveSrc,
                      kind: 'video',
                      name: frame ? `Frame ${frame.name}` : 'input',
                    })
                  }
                  className={`h-full w-full ${mediaFit}`}
                />
              ) : cur.kind === 'audio' ? (
                <div className="flex h-full w-full items-center px-2">
                  <Waveform url={cur.waveform ?? null} className="h-1/2 w-full text-emerald-400" />
                </div>
              ) : (
                <img
                  src={cur.url}
                  alt=""
                  onLoad={(e) => {
                    const img = e.currentTarget
                    if (img.naturalWidth && img.naturalHeight)
                      setAspect(img.naturalWidth / img.naturalHeight)
                  }}
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: cur.saveSrc,
                      name: frame ? `Frame ${frame.name}` : 'input',
                      kind: 'image',
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: cur.saveSrc,
                      kind: 'image',
                      name: frame ? `Frame ${frame.name}` : 'input',
                    })
                  }
                  className={`h-full w-full ${mediaFit}`}
                />
              )
            ) : isLoader ? (
              <div className="flex flex-col items-center gap-2 p-3 text-center">
                <span className="text-[11px] text-zinc-500">
                  Drag an image or video here, from the Library or your computer.
                </span>
                <button
                  onClick={selectLocal}
                  className="nodrag flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1 text-[11px] text-zinc-200 hover:border-zinc-500 hover:text-white"
                >
                  <UploadIcon className="h-3.5 w-3.5" />
                  Select from Local
                </button>
              </div>
            ) : (
              <span className="p-3 text-center text-[11px] text-zinc-600">
                Drop an asset here, or connect a Preview&apos;s output to set this frame&apos;s
                input.
              </span>
            )}

            {count > 1 && (
              <>
                {/* Thumbnail strip - click an input to show it as this frame's main media. */}
                <ThumbStrip
                  items={thumbs.map((t) => ({
                    id: t.id,
                    url: t.url,
                    kind: t.kind,
                    poster: t.poster,
                  }))}
                  selected={safeIdx}
                  onSelect={setIdx}
                  onRemove={(i) => void removeInputById(frameId, thumbs[i].id)}
                />
                {safeIdx === 0 ? (
                  <span className="absolute left-1 top-1 flex items-center gap-1 rounded bg-emerald-500/80 px-1 py-0.5 text-[9px] font-medium text-white">
                    <StarIcon filled className="h-2.5 w-2.5" />
                    Hero
                  </span>
                ) : canReorder ? (
                  <button
                    onClick={makeHero}
                    title="Use this input as the hero"
                    className="nodrag absolute left-1 top-1 flex items-center gap-1 rounded bg-black/60 px-1 py-0.5 text-[9px] text-amber-300 hover:bg-black/80"
                  >
                    <StarIcon className="h-2.5 w-2.5" />
                    Set hero
                  </button>
                ) : null}
              </>
            )}
          </div>

          {/* Loader footer: always-available local picker (also acts as add/replace once media is
              loaded). Generation frames don't get this - their inputs come from the pipeline. */}
          {isLoader && cur && (
            <div className="flex items-center justify-end border-t border-border bg-surface/90 px-2 py-1">
              <button
                onClick={selectLocal}
                title="Add media from your computer"
                className="nodrag flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-black/40 hover:text-white"
              >
                <UploadIcon className="h-3 w-3" />
                Select from Local
              </button>
            </div>
          )}

          {dropActive && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-md border-2 border-dashed border-accent bg-accent/15 text-[11px] font-medium text-panel">
              Add as input
            </div>
          )}
        </div>
      </NodeFrame>

      {/* Data handles with hover hints - Input (emerald, left), Output (indigo, right). */}
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
