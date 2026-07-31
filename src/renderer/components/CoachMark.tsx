/**
 * A numbered callout with a curved arrow pointing at a control on the canvas.
 *
 * Purely presentational: the caller measures the target and passes a point. Only the dismiss button
 * takes pointer events, so the canvas stays usable behind the hint.
 */
export interface Point {
  x: number
  y: number
}

export function CoachMark({
  target,
  step,
  title,
  body,
  side = 'left',
  onDismiss,
}: {
  /** Where the arrow lands, in coordinates relative to the positioned parent. */
  target: Point
  step: number
  title: string
  body: string
  /** Which side of the target the card sits on. */
  side?: 'left' | 'right'
  onDismiss?: () => void
}): React.JSX.Element {
  const gap = 120
  const card = { x: side === 'left' ? target.x - gap : target.x + gap, y: target.y - 78 }
  // A single quadratic curve reads as a gesture rather than a connector line, which matters when it
  // is drawn over a canvas that is itself full of straight edges.
  const from = { x: card.x, y: card.y + 46 }
  const control = { x: (from.x + target.x) / 2, y: from.y - 26 }

  return (
    <>
      <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
        <defs>
          <marker
            id={`coach-arrow-${step}`}
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </marker>
        </defs>
        <path
          d={`M ${from.x} ${from.y} Q ${control.x} ${control.y} ${target.x} ${target.y}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeDasharray="4 3"
          markerEnd={`url(#coach-arrow-${step})`}
          className="text-accent"
        />
      </svg>
      <div
        className="absolute w-52 -translate-x-1/2 rounded-lg border border-accent/50 bg-panel/95 p-2.5 shadow-xl backdrop-blur"
        style={{ left: card.x, top: card.y }}
      >
        <div className="flex items-center gap-1.5">
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-accent text-[9px] font-bold text-black">
            {step}
          </span>
          <span className="text-[11px] font-medium text-zinc-100">{title}</span>
        </div>
        <p className="mt-1 text-[10px] leading-relaxed text-zinc-400">{body}</p>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="pointer-events-auto mt-1.5 rounded px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-surface hover:text-zinc-200"
          >
            Got it
          </button>
        )}
      </div>
    </>
  )
}
