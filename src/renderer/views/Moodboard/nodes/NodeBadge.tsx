import { useCallback, type ReactNode } from 'react'
import { useReactFlow } from '@xyflow/react'
import { useMoodboardStore } from '../../../store/moodboardStore'

/**
 * Shared chrome for the canvas node family, matching the Generate node's look: a floating
 * title "pill" pinned just above a node's top-left corner (icon + label), plus the small
 * Lucide-style icons the nodes badge themselves with. Keeping the pill here guarantees every
 * node — Frame, Preview, Image, Trim, Director… — reads as one consistent card design.
 */

/**
 * Drag ONE node by its title chip, independent of the graph selection. Grabbing a node's body moves
 * the whole selected graph; grabbing the chip repositions just that node. A custom pointer drag (the
 * chip is `nodrag`, so React Flow's selection-based multi-drag never starts), persisted on release.
 */
function useChipDrag(id: string): (e: React.PointerEvent) => void {
  const rf = useReactFlow()
  const updateItem = useMoodboardStore((s) => s.updateItem)
  return useCallback(
    (e: React.PointerEvent) => {
      if (e.button !== 0) return
      const node = rf.getNode(id)
      if (!node) return
      e.stopPropagation()
      const sx = e.clientX
      const sy = e.clientY
      const ox = node.position.x
      const oy = node.position.y
      let moved = false
      const move = (ev: PointerEvent): void => {
        if (!moved && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 3) return
        moved = true
        const zoom = rf.getViewport().zoom || 1
        const x = ox + (ev.clientX - sx) / zoom
        const y = oy + (ev.clientY - sy) / zoom
        rf.setNodes((nds) => nds.map((n) => (n.id === id ? { ...n, position: { x, y } } : n)))
      }
      const up = (): void => {
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
        if (!moved) return
        const n = rf.getNode(id)
        if (n) void updateItem(id, { x: n.position.x, y: n.position.y })
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
    },
    [id, rf, updateItem],
  )
}

/**
 * Row that holds a node's floating badge(s), pinned above the node's top-left corner. Pass
 * `dragNodeId` (the node's id) to make the chip a per-node drag handle — moving just that node
 * rather than the whole selected graph.
 */
export function NodeBadgeRow({
  children,
  dragNodeId,
}: {
  children: ReactNode
  dragNodeId?: string
}): React.JSX.Element {
  const onPointerDown = useChipDrag(dragNodeId ?? '')
  const base = 'absolute -top-7 left-0 z-10 flex items-center gap-1'
  if (!dragNodeId) return <div className={base}>{children}</div>
  return (
    <div
      className={`${base} nodrag cursor-grab select-none active:cursor-grabbing`}
      onPointerDown={onPointerDown}
      title="Drag to move only this node"
    >
      {children}
    </div>
  )
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

/** Play triangle — a node's Run control. */
export function PlayIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className ?? 'h-4 w-4'}
      aria-hidden="true"
    >
      <path d="M8 5v14l11-7z" />
    </svg>
  )
}

/** Filled square — the Run control turns into this to interrupt an in-progress generation. */
export function StopIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className ?? 'h-4 w-4'}
      aria-hidden="true"
    >
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
    </svg>
  )
}

/**
 * Sliders/adjust mark — opens a node's settings sidebar. Shared by the Generate (fal) node and the
 * Inline Core node so both read identically. Has filled dots, so it's its own svg (not the stroked
 * Icon wrapper).
 */
export function AdjustIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className ?? 'h-4 w-4'}
      aria-hidden="true"
    >
      <line x1="4" y1="8" x2="20" y2="8" />
      <line x1="4" y1="16" x2="20" y2="16" />
      <circle cx="9" cy="8" r="2" fill="currentColor" />
      <circle cx="15" cy="16" r="2" fill="currentColor" />
    </svg>
  )
}

/** Magic wand — a generation model node (Z-Image and other `icon:"wand"` descriptors). */
export function WandIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8 19 13M17.8 6.2 19 5M3 21l9-9M12.2 6.2 11 5" />
    </Icon>
  )
}

/** Box — a loader-style node (`icon:"box"`). */
export function BoxIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M21 8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" />
      <path d="m3.3 7 8.7 5 8.7-5M12 22V12" />
    </Icon>
  )
}

/** Type "T" — a text/prompt-style node (`icon:"type"`). */
export function TypeIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M4 7V4h16v3M9 20h6M12 4v16" />
    </Icon>
  )
}

/** Square — a latent/plumbing node (`icon:"square"`), and the generic fallback glyph. */
export function SquareIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
    </Icon>
  )
}

/** Alert triangle — the "models missing" hint on a model node (opens the model popup). */
export function AlertIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
      <path d="M12 9v4M12 17h.01" />
    </Icon>
  )
}

/** Download tray with a down arrow — a component's download action in the model popup. */
export function DownloadIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
    </Icon>
  )
}

export function UploadIcon({ className }: { className?: string }): React.JSX.Element {
  return (
    <Icon className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </Icon>
  )
}
