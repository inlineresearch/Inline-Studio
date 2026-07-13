import { useEffect, useMemo, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { studio } from '@/lib/studio'
import { resolveMedia } from '@/lib/media'
import type { TrimResolved } from '@shared/types'
import { NodeFrame } from './NodeFrame'
import { NodeBadge, NodeBadgeRow, ScissorsIcon } from './NodeBadge'
import { Waveform } from '../../../components/Waveform'
import { useMoodboardStore } from '../../../store/moodboardStore'

const MIN_GAP = 0.1 // seconds — keep the in/out handles from crossing

const fmt = (s: number): string => `${s.toFixed(1)}s`

/**
 * "Edit Video/Audio" (trim) node: 1 input + 1 output. Shows the connected video/audio in
 * a timeline view (filmstrip / waveform) with two draggable in/out handles; the trimmed
 * window is what flows downstream (e.g. into a director). The window persists on the item.
 */
export function TrimNode({ id, selected }: NodeProps): React.JSX.Element {
  const connectors = useMoodboardStore((s) => s.connectors)
  const item = useMoodboardStore((s) => s.items.find((it) => it.id === id))
  const updateItem = useMoodboardStore((s) => s.updateItem)

  // Signature of the single input connection — re-resolve when it changes.
  const inputSig = useMemo(() => {
    const c = connectors.find((k) => k.toItemId === id)
    return c ? c.fromItemId : ''
  }, [connectors, id])

  const [resolved, setResolved] = useState<TrimResolved | null>(null)
  useEffect(() => {
    let cancelled = false
    if (!inputSig) {
      setResolved(null)
      return
    }
    void studio()
      .timeline.resolveTrim(id)
      .then((res) => {
        if (!cancelled) setResolved(res.ok ? res.value : null)
      })
    return () => {
      cancelled = true
    }
  }, [id, inputSig])

  // Effective duration: probed, falling back to the media element's metadata.
  const [elDuration, setElDuration] = useState(0)
  const duration = Math.max(resolved?.durationSec ?? 0, elDuration)

  // Trim window — seeded from the persisted item, defaulting to the full clip.
  const stored = item?.data.trim
  const inPoint = stored ? Math.max(0, stored.inPoint) : 0
  const outPoint =
    stored && stored.outPoint > stored.inPoint ? stored.outPoint : duration || stored?.outPoint || 0
  const [draft, setDraft] = useState<{ inPoint: number; outPoint: number } | null>(null)
  const view = draft ?? { inPoint, outPoint }

  // Once the duration is known and no real out-point is stored yet, persist the full span.
  useEffect(() => {
    if (duration > 0 && (!stored || stored.outPoint <= stored.inPoint)) {
      void updateItem(
        id,
        { data: { ...item?.data, trim: { inPoint: 0, outPoint: duration } } },
        false,
      )
    }
  }, [duration, stored, id, item?.data, updateItem])

  const stripRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<'in' | 'out' | null>(null)

  const onPointerDown = (which: 'in' | 'out') => (e: React.PointerEvent) => {
    e.stopPropagation()
    e.preventDefault()
    if (duration <= 0) return
    dragRef.current = which
    setDraft({ inPoint: view.inPoint, outPoint: view.outPoint })
    const onMove = (ev: PointerEvent): void => {
      const rect = stripRef.current?.getBoundingClientRect()
      if (!rect || !dragRef.current) return
      const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width))
      const sec = frac * duration
      setDraft((d) => {
        const cur = d ?? { inPoint: view.inPoint, outPoint: view.outPoint }
        if (dragRef.current === 'in') {
          return { ...cur, inPoint: Math.min(sec, cur.outPoint - MIN_GAP) }
        }
        return { ...cur, outPoint: Math.max(sec, cur.inPoint + MIN_GAP) }
      })
    }
    const onUp = (): void => {
      document.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerup', onUp)
      dragRef.current = null
      setDraft((d) => {
        if (d) {
          void updateItem(
            id,
            { data: { ...item?.data, trim: { inPoint: d.inPoint, outPoint: d.outPoint } } },
            true,
          )
        }
        return null
      })
    }
    document.addEventListener('pointermove', onMove)
    document.addEventListener('pointerup', onUp)
  }

  const leftPct = duration > 0 ? (view.inPoint / duration) * 100 : 0
  const rightPct = duration > 0 ? (view.outPoint / duration) * 100 : 100
  const src = resolved ? resolveMedia(resolved.mediaPath) : null

  // Keep the preview player within the trim window: start at in-point, stop at out-point.
  const clampPlayback = (el: HTMLMediaElement): void => {
    const start = view.inPoint
    const end = view.outPoint
    if (end > start && el.currentTime >= end) {
      el.pause()
      el.currentTime = start
    } else if (el.currentTime < start - 0.05) {
      el.currentTime = start
    }
  }
  const onMeta = (el: HTMLMediaElement): void => {
    setElDuration(el.duration || 0)
    if (view.inPoint > 0) el.currentTime = view.inPoint
  }
  const onPlayClamp = (el: HTMLMediaElement): void => {
    if (el.currentTime < view.inPoint || el.currentTime >= view.outPoint)
      el.currentTime = view.inPoint
  }

  return (
    <>
      {/* Title + trim-window badges — float above the node, matching the Generate node. */}
      <NodeBadgeRow>
        <NodeBadge icon={<ScissorsIcon />} title={resolved ? resolved.label : 'Edit Video/Audio'}>
          {resolved ? resolved.label : 'Edit Video/Audio'}
        </NodeBadge>
        {duration > 0 && (
          <NodeBadge tone="info">
            {fmt(view.inPoint)}–{fmt(view.outPoint)} ({fmt(view.outPoint - view.inPoint)})
          </NodeBadge>
        )}
      </NodeBadgeRow>

      <Handle
        type="target"
        position={Position.Left}
        id="in"
        title="Wire a video or audio output here"
        className="!h-3.5 !w-3.5 !border-2 !border-surface !bg-indigo-400"
      />
      <Handle
        type="source"
        position={Position.Right}
        id="out"
        title="Outputs the trimmed segment"
        className="!h-3.5 !w-3.5 !border-2 !border-surface !bg-amber-400"
      />
      <NodeFrame
        id={id}
        selected={!!selected}
        minWidth={300}
        minHeight={150}
        padded={false}
        subtleSelect
      >
        <div className="flex h-full w-full flex-col text-zinc-300">
          {!resolved ? (
            <div className="flex flex-1 items-center justify-center px-3 text-center text-[11px] text-zinc-500">
              Wire a video or audio output into the input on the left
            </div>
          ) : (
            <div className="flex flex-1 flex-col gap-1 p-2">
              {/* Media preview — sized to 16:9 like the director node's preview. */}
              {src && resolved.kind === 'video' && (
                <div className="relative aspect-video w-full shrink-0">
                  <video
                    src={src}
                    controls
                    onLoadedMetadata={(e) => onMeta(e.currentTarget)}
                    onTimeUpdate={(e) => clampPlayback(e.currentTarget)}
                    onPlay={(e) => onPlayClamp(e.currentTarget)}
                    style={{ colorScheme: 'dark' }}
                    className="nodrag absolute inset-0 h-full w-full object-contain"
                  />
                </div>
              )}
              {src && resolved.kind === 'image' && (
                <div className="relative aspect-video w-full shrink-0">
                  <img src={src} alt="" className="absolute inset-0 h-full w-full object-contain" />
                </div>
              )}

              {/* Timeline strip with in/out handles */}
              <div
                ref={stripRef}
                className="relative h-10 w-full select-none overflow-hidden rounded bg-black/40"
                style={
                  resolved.kind === 'video' && resolved.thumbnail
                    ? {
                        backgroundImage: `url(${resolveMedia(resolved.thumbnail)})`,
                        backgroundSize: 'auto 100%',
                        backgroundRepeat: 'repeat-x',
                      }
                    : undefined
                }
              >
                {resolved.kind === 'audio' && (
                  <Waveform
                    url={resolved.audioPeaks ? resolveMedia(resolved.audioPeaks) : null}
                    className="absolute inset-0 h-full w-full text-emerald-400"
                  />
                )}
                {/* Dim the trimmed-out ends */}
                <div
                  className="pointer-events-none absolute inset-y-0 left-0 bg-black/60"
                  style={{ width: `${leftPct}%` }}
                />
                <div
                  className="pointer-events-none absolute inset-y-0 right-0 bg-black/60"
                  style={{ width: `${100 - rightPct}%` }}
                />
                {/* In/out handles */}
                <div
                  onPointerDown={onPointerDown('in')}
                  className="nodrag absolute inset-y-0 z-10 w-2 -translate-x-1/2 cursor-ew-resize bg-amber-400"
                  style={{ left: `${leftPct}%` }}
                  title="Trim start"
                />
                <div
                  onPointerDown={onPointerDown('out')}
                  className="nodrag absolute inset-y-0 z-10 w-2 -translate-x-1/2 cursor-ew-resize bg-amber-400"
                  style={{ left: `${rightPct}%` }}
                  title="Trim end"
                />
              </div>

              {/* Play bar — below the media/waveform. */}
              {src && resolved.kind === 'audio' && (
                <audio
                  src={src}
                  controls
                  onLoadedMetadata={(e) => onMeta(e.currentTarget)}
                  onTimeUpdate={(e) => clampPlayback(e.currentTarget)}
                  onPlay={(e) => onPlayClamp(e.currentTarget)}
                  className="flat-audio nodrag w-full"
                />
              )}
            </div>
          )}
        </div>
      </NodeFrame>
    </>
  )
}
