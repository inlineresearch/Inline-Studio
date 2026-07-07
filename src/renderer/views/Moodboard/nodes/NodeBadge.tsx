import type { ReactNode } from 'react'

/**
 * Shared chrome for the canvas node family, matching the Generate node's look: a floating
 * title "pill" pinned just above a node's top-left corner (icon + label), plus the small
 * Lucide-style icons the nodes badge themselves with. Keeping the pill here guarantees every
 * node — Frame, Preview, Image, Trim, Director… — reads as one consistent card design.
 */

/** Row that holds a node's floating badge(s), pinned above the node's top-left corner. */
export function NodeBadgeRow({ children }: { children: ReactNode }): React.JSX.Element {
  return <div className="absolute -top-7 left-0 z-10 flex items-center gap-1">{children}</div>
}

/**
 * A floating pill naming a node: an icon + label, styled like the Generate node's title chip.
 * `tone='info'` renders a lighter secondary badge (a count, a duration, a price) sitting beside
 * the title; pass `accent` to colour that info text (e.g. emerald for a price).
 */
export function NodeBadge({
  icon,
  children,
  tone = 'title',
  accent,
  title,
  tooltip,
}: {
  icon?: ReactNode
  children?: ReactNode
  tone?: 'title' | 'info'
  /** Tailwind text-colour class for an info badge (defaults to muted zinc). */
  accent?: string
  title?: string
  /** A styled hover tooltip shown below the badge (supersedes the native `title`). */
  tooltip?: ReactNode
}): React.JSX.Element {
  const pad = children == null ? 'px-1.5' : tone === 'info' ? 'px-2' : 'pl-2 pr-2.5'
  const color = tone === 'info' ? (accent ?? 'text-zinc-400') : 'text-zinc-200'
  const pill = (
    <div
      title={tooltip ? undefined : title}
      className={`flex h-6 items-center gap-1 rounded-full border border-border bg-panel/95 text-[10px] font-medium shadow-sm backdrop-blur ${pad} ${color}`}
    >
      {icon}
      {children != null && <span className="max-w-[160px] truncate">{children}</span>}
    </div>
  )
  if (!tooltip) return pill
  return (
    <div className="group relative">
      {pill}
      <span className="pointer-events-none absolute left-0 top-full z-20 mt-1 hidden w-52 rounded-md border border-border bg-panel/95 px-2 py-1.5 text-[10px] font-normal leading-relaxed text-zinc-300 shadow-lg backdrop-blur group-hover:block">
        {tooltip}
      </span>
    </div>
  )
}

/** Lucide-style stroked icon wrapper — 24×24, inherits colour/size from the caller. */
function Icon({
  className,
  children,
}: {
  className?: string
  children: ReactNode
}): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className ?? 'h-3.5 w-3.5 shrink-0'}
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

/** Film frame — the Frame node. */
export function FilmIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M7 3v18M17 3v18M3 7.5h4M3 12h18M3 16.5h4M17 7.5h4M17 16.5h4" />
    </Icon>
  )
}

/** Eye — the Preview node. */
export function EyeIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </Icon>
  )
}

/** Picture — an Image asset node. */
export function ImageGlyph({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21" />
    </Icon>
  )
}

/** Movie camera — a Video asset node. */
export function VideoGlyph({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="m22 8-6 4 6 4V8Z" />
      <rect x="2" y="6" width="14" height="12" rx="2" ry="2" />
    </Icon>
  )
}

/** Music note — an Audio asset node. */
export function AudioGlyph({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </Icon>
  )
}

/** Scissors — the Edit/Trim node. */
export function ScissorsIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <circle cx="6" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M8.12 8.12 12 12M20 4 8.12 15.88M14.8 14.8 20 20" />
    </Icon>
  )
}

/** Clapperboard — the Director node. */
export function ClapperIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3Z" />
      <path d="m6.2 5.3 3.1 3.9M12.4 3.4l3.1 4" />
      <path d="M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
    </Icon>
  )
}

/** Chain link — the Frame node's "link a ComfyUI workflow" action. */
export function LinkIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </Icon>
  )
}

export function ChevronLeftIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className ?? 'h-4 w-4'}>
      <path d="m15 18-6-6 6-6" />
    </Icon>
  )
}

export function ChevronRightIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className ?? 'h-4 w-4'}>
      <path d="m9 18 6-6-6-6" />
    </Icon>
  )
}

export function ChevronDownIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className ?? 'h-3 w-3'}>
      <path d="m6 9 6 6 6-6" />
    </Icon>
  )
}

/** Star — "set as hero" / hero marker. `filled` paints it solid for the current hero. */
export function StarIcon({
  className,
  filled,
}: {
  className?: string
  filled?: boolean
}): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className ?? 'h-3 w-3 shrink-0'}
      aria-hidden="true"
    >
      <polygon points="12 2 15.1 8.3 22 9.3 17 14.1 18.2 21 12 17.8 5.8 21 7 14.1 2 9.3 8.9 8.3 12 2" />
    </svg>
  )
}

/** X — close / delete. */
export function XIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className ?? 'h-3.5 w-3.5'}>
      <path d="M18 6 6 18M6 6l12 12" />
    </Icon>
  )
}
