import { useEffect, useMemo, useState } from 'react'

/** Shape written by the main process (electron/main/media/peaks.ts). */
interface PeaksData {
  version: number
  duration: number
  peaks: number[]
}

interface WaveformProps {
  /** URL of the peaks JSON (built with mediaUrl), or null while none exists yet. */
  url: string | null
  /** Playback progress 0..1 - fills bars up to the playhead. */
  progress?: number
  /** Target number of bars to draw; peaks are max-pooled down to this. */
  bars?: number
  className?: string
  /** Click-to-seek: receives a 0..1 fraction of the waveform width. */
  onSeek?: (fraction: number) => void
  /** Render only this fraction [start, end] of the source peaks (e.g. a trimmed window). */
  rangeStart?: number
  rangeEnd?: number
}

/** Max-pool a peaks array down to at most `target` bars. */
function resample(peaks: number[], target: number): number[] {
  if (peaks.length <= target) return peaks
  const size = Math.ceil(peaks.length / target)
  const out: number[] = []
  for (let i = 0; i < peaks.length; i += size) {
    let max = 0
    const end = Math.min(i + size, peaks.length)
    for (let j = i; j < end; j++) if (peaks[j] > max) max = peaks[j]
    out.push(max)
  }
  return out
}

/**
 * Renders an audio waveform from a peaks JSON as an SVG bar chart, with an optional
 * playhead fill. Shows a flat baseline while the waveform is missing/loading (the
 * peaks file is generated in the background after import).
 */
export function Waveform({
  url,
  progress = 0,
  bars = 200,
  className,
  onSeek,
  rangeStart = 0,
  rangeEnd = 1,
}: WaveformProps): React.JSX.Element {
  const [peaks, setPeaks] = useState<number[] | null>(null)

  useEffect(() => {
    if (!url) {
      setPeaks(null)
      return
    }
    let cancelled = false
    fetch(url)
      .then((r) => (r.ok ? (r.json() as Promise<PeaksData>) : null))
      .then((d) => {
        if (!cancelled) setPeaks(d?.peaks ?? null)
      })
      .catch(() => {
        if (!cancelled) setPeaks(null)
      })
    return () => {
      cancelled = true
    }
  }, [url])

  const sampled = useMemo(() => {
    if (!peaks) return null
    // Slice to the requested window (a trimmed clip shows only its in/out portion).
    const lo = Math.max(0, Math.min(1, rangeStart))
    const hi = Math.max(lo, Math.min(1, rangeEnd))
    const arr =
      lo > 0 || hi < 1
        ? peaks.slice(Math.floor(lo * peaks.length), Math.max(1, Math.ceil(hi * peaks.length)))
        : peaks
    return resample(arr, bars)
  }, [peaks, bars, rangeStart, rangeEnd])

  const handleClick = onSeek
    ? (e: React.MouseEvent<SVGSVGElement>) => {
        const rect = e.currentTarget.getBoundingClientRect()
        onSeek(Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)))
      }
    : undefined

  if (!sampled || sampled.length === 0) {
    // Baseline placeholder until peaks exist.
    return (
      <svg
        className={className}
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        onClick={handleClick}
      >
        <line x1="0" y1="50" x2="100" y2="50" stroke="currentColor" strokeWidth="1" opacity="0.3" />
      </svg>
    )
  }

  const n = sampled.length
  const playedTo = progress * n

  return (
    <svg
      className={className}
      viewBox={`0 0 ${n} 100`}
      preserveAspectRatio="none"
      onClick={handleClick}
      style={onSeek ? { cursor: 'pointer' } : undefined}
    >
      {sampled.map((p, i) => {
        const h = Math.max(1, p * 100)
        const y = (100 - h) / 2
        return (
          <rect
            key={i}
            x={i + 0.1}
            y={y}
            width={0.8}
            height={h}
            fill="currentColor"
            opacity={i < playedTo ? 1 : 0.4}
          />
        )
      })}
    </svg>
  )
}
