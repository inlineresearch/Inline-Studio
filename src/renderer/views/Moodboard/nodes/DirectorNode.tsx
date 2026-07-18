import { useEffect, useMemo, useRef, useState } from 'react'
import { Handle, Position, useReactFlow, type NodeProps } from '@xyflow/react'
import type { DirectorClip } from '@shared/types'
import { NodeFrame } from './NodeFrame'
import { ClapperIcon, NodeBadge, NodeBadgeRow } from './NodeBadge'
import { Waveform } from '../../../components/Waveform'
import { useMoodboardStore } from '../../../store/moodboardStore'
import { useTimelineStore } from '../../../store/timelineStore'
import type { DirectorNodeData } from './nodeData'
import { resolveMedia } from '@/lib/media'

const VIDEO_PREFIX = 'vin-'
const AUDIO_PREFIX = 'ain-'
const MIN_VIDEO_INPUTS = 5
const MIN_AUDIO_INPUTS = 2
const REBUILD_DEBOUNCE_MS = 800

const slot = (handle: string | undefined, prefix: string): number | null => {
  if (typeof handle !== 'string' || !handle.startsWith(prefix)) return null
  const n = Number(handle.slice(prefix.length))
  return Number.isFinite(n) ? n : null
}

/**
 * A connection dot living inside one of the node's side gutters (not on the border): an
 * inline React Flow Handle + a tiny label. `position` drives the edge approach direction.
 */
function GutterDot({
  handleId,
  type,
  position,
  label,
  dotClass,
}: {
  handleId: string
  type: 'target' | 'source'
  position: Position
  label: string
  dotClass: string
}): React.JSX.Element {
  return (
    <div className="flex flex-col items-center gap-px" title={label}>
      <Handle
        type={type}
        position={position}
        id={handleId}
        style={{ position: 'relative', left: 0, top: 0, transform: 'none' }}
        className={`!h-3.5 !w-3.5 !min-h-0 !min-w-0 !border-2 !border-surface ${dotClass}`}
      />
      <span className="text-[11px] font-medium leading-none text-zinc-300">{label}</span>
    </div>
  )
}

/**
 * A self-contained video-director node: a mini-NLE on the canvas. Video clips and the L1
 * audio layer are derived from the videos wired into the `vin-*` inputs; the L2 layer is
 * the user's audio wired into the `ain-*` inputs. Inputs grow as they fill. The proxy
 * preview auto-rebuilds (debounced) as connections/volumes change.
 */
export function DirectorNode({ id, data, selected }: NodeProps): React.JSX.Element {
  const { name, previewUrl } = data as DirectorNodeData
  const connectors = useMoodboardStore((s) => s.connectors)
  const items = useMoodboardStore((s) => s.items)
  const reloadBoard = useMoodboardStore((s) => s.load)
  const setConnectorVolume = useMoodboardStore((s) => s.setConnectorVolume)
  const { setCenter } = useReactFlow()
  const timeline = useTimelineStore((s) => s.timelineByOwner[id])
  const progress = useTimelineStore((s) => s.progressByOwner[id])
  const resolve = useTimelineStore((s) => s.resolve)
  const setVolumes = useTimelineStore((s) => s.setVolumes)
  const buildPreview = useTimelineStore((s) => s.buildPreview)
  const exportTimeline = useTimelineStore((s) => s.exportTimeline)

  const incoming = useMemo(() => connectors.filter((c) => c.toItemId === id), [connectors, id])
  // A signature of the wired inputs - drives re-resolve and rebuild. Includes any upstream
  // trim node's in/out window so editing a trim re-resolves this director's timeline.
  const connSig = useMemo(
    () =>
      incoming
        .map((c) => {
          const from = items.find((it) => it.id === c.fromItemId)
          const t = from?.type === 'trim' ? from.data.trim : undefined
          const trimSig = t ? `${t.inPoint},${t.outPoint}` : ''
          const vol = typeof c.data?.volume === 'number' ? c.data.volume : ''
          return `${c.fromItemId}:${String(c.data?.targetHandle ?? '')}:${trimSig}:${vol}`
        })
        .sort()
        .join('|'),
    [incoming, items],
  )

  const usedVideo = new Set(
    incoming
      .map((c) => slot(c.data?.targetHandle as string, VIDEO_PREFIX))
      .filter((n) => n !== null),
  )
  const usedAudio = new Set(
    incoming
      .map((c) => slot(c.data?.targetHandle as string, AUDIO_PREFIX))
      .filter((n) => n !== null),
  )
  const videoHandles = Math.max(MIN_VIDEO_INPUTS, usedVideo.size + 1)
  const audioHandles = Math.max(MIN_AUDIO_INPUTS, usedAudio.size + 1)

  const l1 = timeline?.l1Volume ?? 1
  const l2 = timeline?.l2Volume ?? 1
  const video = timeline?.video ?? []
  const l2clips = timeline?.l2 ?? []
  const hasClips = video.length > 0 || l2clips.length > 0
  const total = Math.max(layerEnd(video), layerEnd(l2clips), 0.001)

  // Re-resolve the derived timeline whenever the wired inputs change.
  useEffect(() => {
    void resolve(id)
  }, [id, connSig, resolve])

  // Debounced proxy rebuild when inputs or volumes change. Any input (video OR audio)
  // is enough - an audio-only timeline renders over a black frame.
  const hasInput = incoming.some((c) => {
    const h = String(c.data?.targetHandle ?? '')
    return h.startsWith(VIDEO_PREFIX) || h.startsWith(AUDIO_PREFIX)
  })
  // Skip the rebuild on the INITIAL mount (e.g. an app restart). Rebuilding every
  // director's proxy on load calls `reloadBoard()` - a full board reload - which races
  // the canvas's node measurement and makes preview nodes collapse/duplicate. Only
  // rebuild when the wired inputs or volumes actually change during the session; the
  // proxy from the previous session is still valid until then.
  const rebuildSig = `${connSig}|${l1}|${l2}|${hasInput ? 1 : 0}`
  const lastRebuildSig = useRef<string | null>(null)
  useEffect(() => {
    if (lastRebuildSig.current === null) {
      lastRebuildSig.current = rebuildSig // baseline the mount state; don't rebuild
      return
    }
    if (lastRebuildSig.current === rebuildSig) return
    lastRebuildSig.current = rebuildSig
    if (!hasInput) return
    const h = setTimeout(async () => {
      const ok = await buildPreview(id)
      if (ok) await reloadBoard()
    }, REBUILD_DEBOUNCE_MS)
    return () => clearTimeout(h)
  }, [id, rebuildSig, hasInput, buildPreview, reloadBoard])

  const rendering = progress !== null && progress !== undefined

  // Pan/zoom the canvas to a clip's source frame node (its "Frame X" tag click).
  const navigateToFrame = (frameId: string): void => {
    const it = items.find((i) => i.type === 'frame' && i.frameId === frameId)
    if (it) void setCenter(it.x + it.width / 2, it.y + it.height / 2, { zoom: 1, duration: 400 })
  }

  // Preview playhead synced to the proxy player.
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playhead, setPlayhead] = useState(0)

  return (
    <>
      {/* Title badge - floats above the node, matching the Generate node. */}
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<ClapperIcon />} title={name || 'Director'}>
          {name || 'Director'}
        </NodeBadge>
      </NodeBadgeRow>

      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={360}
        minHeight={320}
        padded={false}
        subtleSelect
      >
        <div className="flex h-full w-full text-zinc-300">
          {/* Left gutter - a dedicated side panel for the input dots (not on the border). */}
          <div className="flex w-11 shrink-0 flex-col items-center gap-1.5 overflow-y-auto border-r border-border bg-panel py-1">
            {Array.from({ length: videoHandles }).map((_, i) => (
              <GutterDot
                key={`vin-${i}`}
                handleId={`${VIDEO_PREFIX}${i}`}
                type="target"
                position={Position.Left}
                label={`V${i + 1}`}
                dotClass="!bg-indigo-400"
              />
            ))}
            {Array.from({ length: audioHandles }).map((_, i) => (
              <GutterDot
                key={`ain-${i}`}
                handleId={`${AUDIO_PREFIX}${i}`}
                type="target"
                position={Position.Left}
                label={`A${i + 1}`}
                dotClass="!bg-emerald-400"
              />
            ))}
          </div>

          {/* Center column */}
          <div className="flex min-w-0 flex-1 flex-col">
            {/* Header - a controls bar (the title lives in the floating badge above). */}
            <div className="flex shrink-0 items-center justify-end gap-1 border-b border-border bg-panel px-2 py-1">
              {rendering && (
                <span className="mr-auto text-[10px] text-amber-400">
                  {Math.round((progress ?? 0) * 100)}%
                </span>
              )}
              <button
                onClick={async () => {
                  const path = await exportTimeline(id)
                  if (path) await reloadBoard()
                }}
                disabled={rendering || !hasClips}
                className="nodrag rounded bg-accent px-1.5 py-0.5 text-[10px] text-panel hover:brightness-110 disabled:opacity-40"
              >
                Export
              </button>
              {/* Output link, after the Export button. */}
              <span
                className="flex items-center gap-1 text-[11px] font-medium text-zinc-300"
                title="Video output"
              >
                Out
                <Handle
                  type="source"
                  position={Position.Right}
                  id="out"
                  style={{ position: 'relative', left: 0, top: 0, transform: 'none' }}
                  className="!h-3.5 !w-3.5 !min-h-0 !min-w-0 !border-2 !border-surface !bg-amber-400"
                />
              </span>
            </div>

            {/* Preview - sized to the 16:9 proxy so the video fills it (no black bars). */}
            <div className="relative aspect-video w-full shrink-0 bg-black">
              {previewUrl ? (
                <video
                  ref={videoRef}
                  src={resolveMedia(previewUrl)}
                  controls
                  onTimeUpdate={() => {
                    const v = videoRef.current
                    if (v && v.duration > 0) setPlayhead(v.currentTime / v.duration)
                  }}
                  className="nodrag absolute inset-0 h-full w-full object-contain"
                />
              ) : (
                <span className="absolute inset-0 flex items-center justify-center px-3 text-center text-[11px] text-zinc-500">
                  {!hasClips
                    ? 'Wire frames into the V (video) or A (audio) inputs on the left'
                    : rendering
                      ? 'Rendering preview…'
                      : 'Building preview…'}
                </span>
              )}
            </div>

            {/* Tracks */}
            <div className="nodrag relative flex-1 overflow-auto py-1.5">
              {/* Shared playhead across the tracks (aligned to the track lanes' inner width). */}
              {hasClips && (
                <div
                  className="pointer-events-none absolute inset-y-0 z-10 w-px bg-red-500"
                  style={{ left: `${(playhead * 100).toFixed(2)}%` }}
                />
              )}

              {hasClips && <Ruler total={total} />}

              <TrackRow label="VIDEO">
                {video.map((c) => (
                  <ClipBlock
                    key={c.key}
                    clip={c}
                    total={total}
                    kind="video"
                    onNavigate={navigateToFrame}
                  />
                ))}
              </TrackRow>

              <TrackRow
                label="AUDIO L1"
                control={<VolumeSlider value={l1} onChange={(v) => void setVolumes(id, v, l2)} />}
              >
                {video.map((c) => (
                  <ClipBlock
                    key={c.key}
                    clip={c}
                    total={total}
                    kind="audio"
                    onSetVolume={(v) => void setConnectorVolume(c.connectorId, v)}
                  />
                ))}
              </TrackRow>

              <TrackRow
                label="AUDIO L2"
                control={<VolumeSlider value={l2} onChange={(v) => void setVolumes(id, l1, v)} />}
              >
                {l2clips.length === 0 ? (
                  <span className="px-1 text-[9px] text-zinc-600">
                    Wire audio into a green input
                  </span>
                ) : (
                  l2clips.map((c) => <ClipBlock key={c.key} clip={c} total={total} kind="audio" />)
                )}
              </TrackRow>
            </div>
          </div>
        </div>
      </NodeFrame>
    </>
  )
}

function layerEnd(clips: DirectorClip[]): number {
  return clips.reduce((m, c) => Math.max(m, c.startTime + c.duration), 0)
}

function TrackRow({
  label,
  control,
  children,
}: {
  label: string
  control?: React.ReactNode
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="mb-1.5">
      <div className="mb-0.5 flex items-center justify-between">
        <span className="text-[9px] uppercase tracking-wide text-zinc-500">{label}</span>
        {control}
      </div>
      <div className="flex h-8 w-full overflow-hidden rounded bg-black/30">{children}</div>
    </div>
  )
}

function ClipBlock({
  clip,
  total,
  kind,
  onNavigate,
  onSetVolume,
}: {
  clip: DirectorClip
  total: number
  kind: 'video' | 'audio'
  /** Pan the canvas to the clip's source frame (only for frame-backed clips). */
  onNavigate?: (frameId: string) => void
  /** Per-input volume control (L1 audio clips). */
  onSetVolume?: (volume: number) => void
}): React.JSX.Element {
  const width = `${(clip.duration / total) * 100}%`
  if (kind === 'audio') {
    return (
      <div
        style={{ width }}
        className="relative flex h-full items-center border-r border-black/40 bg-emerald-500/10 px-0.5"
        title={`${clip.label} (${clip.duration.toFixed(1)}s)`}
      >
        <Waveform
          url={clip.audioPeaks ? resolveMedia(clip.audioPeaks) : null}
          rangeStart={clip.peaksStart}
          rangeEnd={clip.peaksEnd}
          className="h-2/3 w-full text-emerald-400"
        />
        {onSetVolume && <ClipVolume volume={clip.volume} onChange={onSetVolume} />}
      </div>
    )
  }

  // Video/image clip: a filmstrip (video) or still (image) background, with a "Frame X" tag.
  // Filmstrips repeat horizontally at full height; stills cover the clip.
  const bg: React.CSSProperties = clip.thumbnail
    ? {
        width,
        backgroundImage: `url(${resolveMedia(clip.thumbnail)})`,
        backgroundSize: clip.kind === 'video' ? 'auto 100%' : 'cover',
        backgroundRepeat: clip.kind === 'video' ? 'repeat-x' : 'no-repeat',
        backgroundPosition: 'center',
      }
    : { width }

  return (
    <div
      style={bg}
      className={`relative h-full overflow-hidden border-r border-black/40 ${clip.thumbnail ? 'bg-black/40' : 'bg-indigo-500/30'}`}
      title={`${clip.label} (${clip.duration.toFixed(1)}s)`}
    >
      <button
        onClick={(e) => {
          e.stopPropagation()
          if (clip.frameId) onNavigate?.(clip.frameId)
        }}
        disabled={!clip.frameId}
        title={clip.frameId ? `Go to Frame ${clip.label}` : clip.label}
        className="nodrag absolute left-0.5 top-0.5 max-w-[calc(100%-4px)] truncate rounded bg-black/70 px-1 text-[9px] text-indigo-100 hover:bg-black/90 disabled:cursor-default"
      >
        Frame {clip.label}
      </button>
    </div>
  )
}

/** A seconds ruler aligned to the track lanes (0 → total). */
function Ruler({ total }: { total: number }): React.JSX.Element {
  const step = total <= 10 ? 1 : total <= 30 ? 2 : total <= 60 ? 5 : total <= 180 ? 10 : 30
  const ticks: number[] = []
  for (let t = step; t < total - step / 2; t += step) ticks.push(t)
  return (
    <div className="relative mb-1 h-3 w-full border-b border-border/40">
      <span className="absolute left-0 top-0 text-[8px] text-zinc-500">0s</span>
      {ticks.map((t) => (
        <span
          key={t}
          className="absolute top-0 -translate-x-1/2 text-[8px] text-zinc-500"
          style={{ left: `${(t / total) * 100}%` }}
        >
          {t}s
        </span>
      ))}
      <span className="absolute right-0 top-0 text-[8px] text-zinc-500">
        {total < 10 ? total.toFixed(1) : Math.round(total)}s
      </span>
    </div>
  )
}

/** A line volume icon (Lucide-style), matching the widget icons. */
function VolumeIcon({
  muted,
  className,
}: {
  muted?: boolean
  className?: string
}): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className ?? 'h-3 w-3'}
    >
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      {muted ? (
        <>
          <line x1="22" y1="9" x2="16" y2="15" />
          <line x1="16" y1="9" x2="22" y2="15" />
        </>
      ) : (
        <>
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        </>
      )}
    </svg>
  )
}

/** Per-input volume: a speaker icon that toggles an inline slider over the clip. */
function ClipVolume({
  volume,
  onChange,
}: {
  volume: number
  onChange: (v: number) => void
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        onClick={(e) => {
          e.stopPropagation()
          setOpen((o) => !o)
        }}
        title={`Volume ${Math.round(volume * 100)}%`}
        className="nodrag absolute left-0.5 top-0.5 z-10 flex h-4 w-4 items-center justify-center rounded bg-black/60 text-zinc-100 hover:bg-black/80"
      >
        <VolumeIcon muted={volume <= 0} className="h-2.5 w-2.5" />
      </button>
      {open && (
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(volume * 100)}
          onChange={(e) => onChange(Number(e.target.value) / 100)}
          onPointerDown={(e) => e.stopPropagation()}
          className="nodrag absolute inset-x-1 bottom-0.5 z-10 h-1 accent-emerald-400"
        />
      )}
    </>
  )
}

function VolumeSlider({
  value,
  onChange,
}: {
  value: number
  onChange: (v: number) => void
}): React.JSX.Element {
  return (
    <label className="flex items-center gap-1" title={`Volume ${Math.round(value * 100)}%`}>
      <span className="text-zinc-400">
        <VolumeIcon muted={value <= 0} className="h-3 w-3" />
      </span>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(value * 100)}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="nodrag h-1 w-16 accent-emerald-400"
      />
    </label>
  )
}
