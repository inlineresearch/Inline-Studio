/**
 * The "Add node" popup opened from the toolbar's + button, or by double-clicking empty canvas in
 * Select mode. Lists every node type. The single "Generate" row creates a unified generation frame
 * (a chooser — Link a ComfyUI Workflow or Generate with Fal API — resolved on the node itself).
 * Positioning mirrors MoodboardPanel's "Connect to…" menu (container-relative left/top).
 */

/** The node kinds the Add menu can create (Text has its own toolbar tool, so it's not here). */
export type AddNodeKind = 'frame' | 'layer' | 'preview' | 'director' | 'trim' | 'prompt'

interface Entry {
  kind: AddNodeKind
  label: string
  icon: React.JSX.Element
  /** Accent the row (the generation node is the AI action). */
  accent?: boolean
}

const ENTRIES: Entry[] = [
  { kind: 'frame', label: 'Generate', icon: <SparklesIcon />, accent: true },
  { kind: 'layer', label: 'Layer', icon: <LayerIcon /> },
  { kind: 'preview', label: 'Preview', icon: <ImageIcon /> },
  { kind: 'director', label: 'Video Director', icon: <ClapperboardIcon /> },
  { kind: 'trim', label: 'Edit Video/Audio', icon: <ScissorsIcon /> },
  { kind: 'prompt', label: 'Prompt', icon: <PromptIcon /> },
]

export function AddNodeMenu({
  x,
  y,
  above = false,
  onPick,
  onClose,
}: {
  /** Container-relative anchor point (px). */
  x: number
  y: number
  /** When true the menu is centered on x and grows upward from y (used by the toolbar button). */
  above?: boolean
  onPick: (kind: AddNodeKind) => void
  onClose: () => void
}): React.JSX.Element {
  return (
    <>
      <div className="absolute inset-0 z-20" onClick={onClose} />
      <div
        className={`absolute z-30 flex w-64 select-none flex-col overflow-hidden rounded-md border border-border bg-panel text-xs shadow-xl ${
          above ? '-translate-x-1/2 -translate-y-full' : ''
        }`}
        style={{ left: x, top: y }}
      >
        <div className="border-b border-border px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">
          Add node
        </div>
        {ENTRIES.map((e) => (
          <button
            key={e.kind}
            onClick={() => onPick(e.kind)}
            className={`flex items-center gap-2 px-2.5 py-1.5 text-left hover:bg-surface ${
              e.accent ? 'text-emerald-300' : 'text-zinc-200'
            }`}
          >
            <span className="flex h-4 w-4 shrink-0 items-center justify-center">{e.icon}</span>
            {e.label}
          </button>
        ))}
      </div>
    </>
  )
}

// ── Node icons (moved here from CanvasToolbar, which no longer shows per-node buttons) ──

function LayerIcon(): React.JSX.Element {
  return (
    <Svg>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </Svg>
  )
}

function ImageIcon(): React.JSX.Element {
  return (
    <Svg>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="M21 15l-5-5L5 21" />
    </Svg>
  )
}

function ClapperboardIcon(): React.JSX.Element {
  return (
    <Svg>
      <path d="M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3Z" />
      <path d="m6.2 5.3 3.1 3.9" />
      <path d="m12.4 3.4 3.1 4" />
      <path d="M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
      <path d="M8 15.5h8" />
      <path d="M8 18.5h6" />
    </Svg>
  )
}

function ScissorsIcon(): React.JSX.Element {
  return (
    <Svg>
      <circle cx="6" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M20 4 8.12 15.88M14.47 14.48 20 20M8.12 8.12 12 12" />
    </Svg>
  )
}

function SparklesIcon(): React.JSX.Element {
  return (
    <Svg>
      <path d="M12 3l1.9 4.8L18.6 9.7 13.9 11.6 12 16.4 10.1 11.6 5.4 9.7 10.1 7.8Z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z" />
    </Svg>
  )
}

function PromptIcon(): React.JSX.Element {
  return (
    <Svg>
      <path d="M4 5h16M4 5v14M4 12h10" />
      <path d="M15 19l2 2 4-4" />
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
      className="h-4 w-4"
    >
      {children}
    </svg>
  )
}
