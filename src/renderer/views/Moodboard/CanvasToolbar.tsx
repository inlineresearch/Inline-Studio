/**
 * Floating bottom-center widget bar: a horizontal strip of canvas tools - Select (edit), Pan
 * (view), Text, and Add (opens a node list). It sits inside the canvas area, so it tracks the
 * Assets panel as it opens/resizes.
 *
 * Shared by both canvases. The Trainer omits `onAddText`, since it has no text node, and the
 * button is dropped rather than shown disabled.
 */
export function CanvasToolbar({
  tool,
  onSelectTool,
  onPanTool,
  onAddText,
  onOpenAdd,
}: {
  /** The active interaction tool. */
  tool: 'select' | 'pan'
  onSelectTool: () => void
  onPanTool: () => void
  /** Add a text node (at canvas centre). Omitted on canvases without text nodes (the Trainer). */
  onAddText?: () => void
  /** Open the "Add node" list, anchored to the + button. */
  onOpenAdd: (buttonRect: DOMRect) => void
}): React.JSX.Element {
  return (
    <div className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 flex-row items-center gap-1 rounded-lg border border-border bg-panel/95 p-1 shadow-lg backdrop-blur">
      <ToolButton label="Select (edit)" active={tool === 'select'} onClick={onSelectTool}>
        <CursorIcon />
      </ToolButton>
      <ToolButton label="Pan (view)" active={tool === 'pan'} onClick={onPanTool}>
        <HandIcon />
      </ToolButton>
      <div className="mx-0.5 h-6 w-px self-center bg-border" />
      {onAddText && (
        <ToolButton label="Add text" onClick={onAddText}>
          <span className="text-base font-bold leading-none">T</span>
        </ToolButton>
      )}
      <ToolButton
        label="Add node"
        onClick={(e) => onOpenAdd(e.currentTarget.getBoundingClientRect())}
      >
        <PlusIcon />
      </ToolButton>
    </div>
  )
}

function ToolButton({
  label,
  active = false,
  onClick,
  children,
}: {
  label: string
  active?: boolean
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <button
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={`group relative flex h-9 w-9 items-center justify-center rounded-md transition-colors ${
        active ? 'bg-surface text-accent' : 'text-accent hover:bg-surface hover:brightness-110'
      }`}
    >
      {children}
      {/* Hover label sits above the button, since the bar hugs the bottom edge. */}
      <span className="pointer-events-none absolute bottom-full left-1/2 mb-2 -translate-x-1/2 whitespace-nowrap rounded bg-black/80 px-1.5 py-0.5 text-[11px] text-zinc-100 opacity-0 shadow transition-opacity group-hover:opacity-100">
        {label}
      </span>
    </button>
  )
}

function CursorIcon(): React.JSX.Element {
  return (
    <Svg>
      <path d="m3 3 7.5 18 2.6-7.9L21 10.5 3 3z" />
      <path d="m13 13 6 6" />
    </Svg>
  )
}

function HandIcon(): React.JSX.Element {
  return (
    <Svg>
      <path d="M18 11V6a2 2 0 0 0-4 0" />
      <path d="M14 10V4a2 2 0 0 0-4 0v2" />
      <path d="M10 10.5V6a2 2 0 0 0-4 0v8" />
      <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-6-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
    </Svg>
  )
}

function PlusIcon(): React.JSX.Element {
  return (
    <Svg>
      <path d="M12 5v14M5 12h14" />
    </Svg>
  )
}

function Svg({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-[18px] w-[18px]"
    >
      {children}
    </svg>
  )
}
