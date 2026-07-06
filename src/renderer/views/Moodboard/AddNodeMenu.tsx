/**
 * The "Add node" popup opened from the toolbar's + button, or by double-clicking empty canvas in
 * Select mode. Lists every node type; the "Generate" row drills into the model list (grouped by
 * provider). Picking creates the node at the caller-supplied flow position. Positioning mirrors
 * MoodboardPanel's "Connect to…" menu (container-relative left/top).
 */
import { useState } from 'react'
import { listNodeDefs, groupByOwner } from '@shared/nodes/registry'

/** The node kinds the Add menu can create (Text has its own toolbar tool, so it's not here). */
export type AddNodeKind =
  | 'frame'
  | 'layer'
  | 'preview'
  | 'director'
  | 'trim'
  | 'generate'
  | 'prompt'

interface Entry {
  kind: AddNodeKind
  label: string
  icon: React.JSX.Element
  /** Accent the row (Generate is the AI action). */
  accent?: boolean
}

const ENTRIES: Entry[] = [
  { kind: 'frame', label: 'Frame', icon: <FrameIcon /> },
  { kind: 'layer', label: 'Layer', icon: <LayerIcon /> },
  { kind: 'preview', label: 'Preview', icon: <ImageIcon /> },
  { kind: 'director', label: 'Video Director', icon: <ClapperboardIcon /> },
  { kind: 'trim', label: 'Edit Video/Audio', icon: <ScissorsIcon /> },
  { kind: 'generate', label: 'Generate', icon: <SparklesIcon />, accent: true },
  { kind: 'prompt', label: 'Prompt', icon: <PromptIcon /> },
]

export function AddNodeMenu({
  x,
  y,
  above = false,
  onPick,
  onPickModel,
  onClose,
}: {
  /** Container-relative anchor point (px). */
  x: number
  y: number
  /** When true the menu is centered on x and grows upward from y (used by the toolbar button). */
  above?: boolean
  onPick: (kind: AddNodeKind) => void
  /** Create a Generate node with a specific model id. */
  onPickModel: (modelId: string) => void
  onClose: () => void
}): React.JSX.Element {
  const [view, setView] = useState<'root' | 'models'>('root')
  return (
    <>
      <div className="absolute inset-0 z-20" onClick={onClose} />
      <div
        className={`absolute z-30 flex w-64 select-none flex-col overflow-hidden rounded-md border border-border bg-panel text-xs shadow-xl ${
          above ? '-translate-x-1/2 -translate-y-full' : ''
        }`}
        style={{ left: x, top: y }}
      >
        {view === 'root' ? (
          <>
            <div className="border-b border-border px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">
              Add node
            </div>
            {ENTRIES.map((e) =>
              e.kind === 'generate' ? (
                <button
                  key={e.kind}
                  onClick={() => setView('models')}
                  className="flex items-center gap-2 px-2.5 py-1.5 text-left text-emerald-300 hover:bg-surface"
                >
                  <span className="flex h-4 w-4 shrink-0 items-center justify-center">
                    {e.icon}
                  </span>
                  <span className="flex-1">{e.label}</span>
                  <ChevronRightIcon />
                </button>
              ) : (
                <button
                  key={e.kind}
                  onClick={() => onPick(e.kind)}
                  className="flex items-center gap-2 px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
                >
                  <span className="flex h-4 w-4 shrink-0 items-center justify-center">
                    {e.icon}
                  </span>
                  {e.label}
                </button>
              ),
            )}
          </>
        ) : (
          <>
            <button
              onClick={() => setView('root')}
              className="flex items-center gap-1 border-b border-border px-2 py-1 text-left text-[10px] uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
            >
              <ChevronLeftIcon /> Generate
            </button>
            <div className="max-h-72 overflow-y-auto py-0.5">
              {groupByOwner(listNodeDefs()).map((group) => (
                <div key={group.owner}>
                  <div className="px-2.5 pb-0.5 pt-1.5 text-[9px] font-semibold uppercase tracking-wide text-zinc-500">
                    {group.label}
                  </div>
                  {group.defs.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => onPickModel(d.id)}
                      className="flex w-full flex-col items-start px-2.5 py-1.5 text-left hover:bg-surface"
                    >
                      <span className="w-full truncate text-[11px] text-zinc-100">{d.title}</span>
                      <span className="text-[10px] text-zinc-500">{d.category}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </>
  )
}

// ── Node icons (moved here from CanvasToolbar, which no longer shows per-node buttons) ──

function FrameIcon(): React.JSX.Element {
  return (
    <Svg>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 9h18M3 15h18M9 4v16" />
    </Svg>
  )
}

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

function ChevronRightIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3 w-3 shrink-0 text-zinc-500"
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  )
}

function ChevronLeftIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3 w-3 shrink-0"
    >
      <path d="m15 18-6-6 6-6" />
    </svg>
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
