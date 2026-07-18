import { useLayoutEffect, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { takeWaveformPath } from '@shared/media'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useFrameStore } from '../../../store/frameStore'
import { useAssetStore } from '../../../store/assetStore'
import { useMediaContextMenu } from '../../../lib/mediaContextMenu'
import { useLightboxStore } from '../../../store/lightboxStore'
import { Waveform } from '../../../components/Waveform'
import { NodeFrame } from './NodeFrame'
import { EyeIcon, NodeBadge, NodeBadgeRow, StarIcon } from './NodeBadge'
import { ThumbStrip } from './ThumbStrip'
import { resolveMedia } from '@/lib/media'

/**
 * A Comfy-style preview node: connect a frame's output handle to its input and it
 * displays that frame's outputs (takes). With several takes it becomes a carousel -
 * page through them and "set hero" to pick the one the timeline points at.
 */
export function PreviewNode({ id, selected }: NodeProps): React.JSX.Element {
  const connectors = useMoodboardStore((s) => s.connectors)
  const items = useMoodboardStore((s) => s.items)
  const item = items.find((it) => it.id === id)
  const updateItem = useMoodboardStore((s) => s.updateItem)
  const frames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)
  const inputsByFrame = useFrameStore((s) => s.inputsByFrame)
  const setHero = useFrameStore((s) => s.setHero)
  const assets = useAssetStore((s) => s.assets)
  const onMediaContextMenu = useMediaContextMenu()
  const openLightbox = useLightboxStore((s) => s.open)
  const [idx, setIdx] = useState(0)
  // Fit the node height to the displayed media's aspect ratio (no black letterbox bars).
  const [aspect, setAspect] = useState<number | null>(null)
  const bodyRef = useRef<HTMLDivElement>(null)

  const conn = connectors.find((c) => c.toItemId === id)
  const sourceItem = conn ? items.find((it) => it.id === conn.fromItemId) : undefined
  const frame = sourceItem?.frameId ? frames.find((s) => s.id === sourceItem.frameId) : undefined
  // A director node's "Out" feeds its exported full-res video (set when Export is clicked).
  const isDirector = sourceItem?.type === 'director'
  const directorExport = isDirector ? (sourceItem?.data.directorExport ?? null) : null

  // Outputs newest-first, but float the hero take to the front so it shows by default.
  const takes = frame ? (takesByFrame[frame.id] ?? []) : []
  const heroId = frame?.heroTakeId ?? null
  const ordered = [...takes]
  if (heroId) {
    const i = ordered.findIndex((t) => t.id === heroId)
    if (i > 0) ordered.unshift(ordered.splice(i, 1)[0])
  }

  const count = ordered.length
  const safeIdx = count ? Math.min(idx, count - 1) : 0
  const cur = count ? ordered[safeIdx] : undefined
  const curIsHero = !!cur && cur.id === heroId

  const makeHero = (): void => {
    if (frame && cur && !curIsHero) void setHero(frame.id, cur.id)
  }

  // When the frame has no takes yet (e.g. an imported frame with no workflow), fall back
  // to its input asset so the preview still shows the contained media.
  const fallbackAsset = (() => {
    if (cur || !frame) return null
    const input = (inputsByFrame[frame.id] ?? []).find((i) => i.assetId)
    return input?.assetId ? (assets.find((a) => a.id === input.assetId) ?? null) : null
  })()

  // Unified media to render: the current take, or the fallback input asset.
  const display = cur
    ? {
        src: resolveMedia(cur.filePath),
        saveSrc: resolveMedia(cur.filePath),
        kind: cur.kind,
        waveform: resolveMedia(takeWaveformPath(cur.id)),
      }
    : fallbackAsset
      ? {
          src: resolveMedia(fallbackAsset.previewPath ?? fallbackAsset.filePath),
          saveSrc: resolveMedia(fallbackAsset.filePath),
          kind: fallbackAsset.kind,
          waveform: fallbackAsset.thumbPath ? resolveMedia(fallbackAsset.thumbPath) : null,
        }
      : null

  // Audio shows a waveform (fixed height); only image/video drive the aspect fit.
  const fitsAspect = isDirector || display?.kind === 'video' || display?.kind === 'image'
  const itemWidth = item?.width
  const itemHeight = item?.height
  // The media body is a CSS aspect-ratio box (like the director's preview), so the video
  // always fills it with no black edges. Here we just size the *node* to hug that box -
  // node height = header height + (width / aspect) - so resizing the width keeps aspect at
  // any size. (Dragging height alone snaps back, i.e. the node maintains the aspect ratio.)
  useLayoutEffect(() => {
    const body = bodyRef.current
    if (!fitsAspect || !aspect || !body || itemWidth == null) return
    const target = Math.round(body.offsetTop + body.offsetWidth / aspect)
    if (itemHeight != null && Math.abs(target - itemHeight) < 1) return
    void updateItem(id, { height: target }, false)
  }, [fitsAspect, aspect, itemWidth, itemHeight, id, updateItem])

  const badgeLabel = `Preview${frame ? ` · Frame ${frame.name}` : isDirector ? ' · Director' : ''}`

  return (
    <>
      {/* Title + take-count badges - float above the node, matching the Generate node. */}
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<EyeIcon />} title={badgeLabel}>
          {badgeLabel}
        </NodeBadge>
        {count > 0 && (
          <NodeBadge tone="info">
            {safeIdx + 1}/{count}
          </NodeBadge>
        )}
      </NodeBadgeRow>

      <Handle
        type="target"
        position={Position.Left}
        id="in"
        className="!h-3.5 !w-3.5 !border-2 !border-surface !bg-indigo-400"
      />
      {/* Output handle: wire to a Frame's input to feed it the selected (hero) take. */}
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        title="Feed the selected output into a frame's input"
        className="!h-3.5 !w-3.5 !border-2 !border-surface !bg-emerald-400"
      />
      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={220}
        minHeight={170}
        padded={false}
        subtleSelect
      >
        <div className="flex h-full w-full flex-col">
          <div
            ref={bodyRef}
            className={`relative w-full overflow-hidden bg-black ${fitsAspect ? '' : 'flex flex-1 items-center justify-center'}`}
            style={fitsAspect ? { aspectRatio: aspect ?? 16 / 9 } : undefined}
          >
            {isDirector ? (
              directorExport ? (
                <video
                  src={resolveMedia(directorExport)}
                  controls
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget
                    if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight)
                  }}
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: resolveMedia(directorExport),
                      name: 'director',
                      kind: 'video',
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: resolveMedia(directorExport),
                      kind: 'video',
                      name: 'director',
                    })
                  }
                  className="absolute inset-0 h-full w-full object-contain"
                />
              ) : (
                <span className="absolute inset-0 flex items-center justify-center p-3 text-center text-[11px] text-zinc-500">
                  Click Export on the director to render the video here.
                </span>
              )
            ) : display ? (
              display.kind === 'video' ? (
                <video
                  src={display.src}
                  controls
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget
                    if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight)
                  }}
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: display.saveSrc,
                      name: frame ? `Frame ${frame.name}` : 'take',
                      kind: 'video',
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: display.saveSrc,
                      kind: 'video',
                      name: frame ? `Frame ${frame.name}` : 'take',
                    })
                  }
                  className="absolute inset-0 h-full w-full object-contain"
                />
              ) : display.kind === 'audio' ? (
                <div className="flex h-full w-full flex-col justify-center gap-2 px-3">
                  <Waveform url={display.waveform} className="h-16 w-full text-emerald-400" />
                  <audio
                    src={display.src}
                    controls
                    onContextMenu={(e) =>
                      onMediaContextMenu(e, {
                        src: display.saveSrc,
                        name: frame ? `Frame ${frame.name}` : 'take',
                        kind: 'audio',
                      })
                    }
                    className="nodrag w-full"
                  />
                </div>
              ) : (
                <img
                  src={display.src}
                  alt=""
                  onLoad={(e) => {
                    const im = e.currentTarget
                    if (im.naturalWidth && im.naturalHeight)
                      setAspect(im.naturalWidth / im.naturalHeight)
                  }}
                  onContextMenu={(e) =>
                    onMediaContextMenu(e, {
                      src: display.saveSrc,
                      name: frame ? `Frame ${frame.name}` : 'take',
                      kind: display.kind,
                    })
                  }
                  onDoubleClick={() =>
                    openLightbox({
                      src: display.saveSrc,
                      kind: 'image',
                      name: frame ? `Frame ${frame.name}` : 'take',
                    })
                  }
                  className="absolute inset-0 h-full w-full object-contain"
                />
              )
            ) : (
              <span className="p-3 text-center text-[11px] text-zinc-500">
                {frame
                  ? 'No outputs yet - generate this frame, or it shows its input.'
                  : "Connect a frame's output here to preview it"}
              </span>
            )}

            {/* Multiple takes → a thumbnail strip; click one to make it the main preview. */}
            <ThumbStrip
              items={ordered.map((t) => ({
                id: t.id,
                url: resolveMedia(t.filePath),
                kind: t.kind,
              }))}
              selected={safeIdx}
              onSelect={setIdx}
            />

            {cur &&
              (curIsHero ? (
                <span className="absolute left-1 top-1 flex items-center gap-1 rounded bg-emerald-500/80 px-1 py-0.5 text-[9px] font-medium text-white">
                  <StarIcon filled className="h-2.5 w-2.5" />
                  Hero
                </span>
              ) : (
                <button
                  onClick={makeHero}
                  title="Use this take as the frame's hero"
                  className="nodrag absolute left-1 top-1 flex items-center gap-1 rounded bg-black/60 px-1 py-0.5 text-[9px] text-amber-300 hover:bg-black/80"
                >
                  <StarIcon className="h-2.5 w-2.5" />
                  Set hero
                </button>
              ))}
          </div>
        </div>
      </NodeFrame>
    </>
  )
}
