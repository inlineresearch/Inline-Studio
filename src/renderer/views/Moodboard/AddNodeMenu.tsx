/**
 * The "Add node" popup opened from the toolbar's + button, or by double-clicking empty canvas in
 * Select mode. Lists the built-in node types plus the Fal Models, Inline Core, and Extension nodes.
 * A fal generation node is added directly from the Fal Models list.
 * Positioning mirrors MoodboardPanel's "Connect to…" menu (container-relative left/top).
 */

import { addableCoreNodes, type NodeDescriptor } from '@shared/coreNodes'
import { isExtensionNode, extensionOf } from '@shared/extensions'
import { listNodeDefs, groupByOwner } from '@shared/nodes/registry'

/** The node kinds the Add menu can create (Text has its own toolbar tool, so it's not here). */
export type AddNodeKind = 'load' | 'layer' | 'preview' | 'director' | 'trim' | 'prompt'

interface Entry {
  kind: AddNodeKind
  label: string
  icon: React.JSX.Element
  /** Accent the row (the generation node is the AI action). */
  accent?: boolean
}

const ENTRIES: Entry[] = [
  { kind: 'load', label: 'Load Assets', icon: <ImageIcon /> },
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
  coreNodes = [],
  onPick,
  onPickCore,
  onPickGen,
  onClose,
}: {
  /** Container-relative anchor point (px). */
  x: number
  y: number
  /** When true the menu is centered on x and grows upward from y (used by the toolbar button). */
  above?: boolean
  /** Inline Core node descriptors (from /v1/models), listed under their categories. */
  coreNodes?: NodeDescriptor[]
  onPick: (kind: AddNodeKind) => void
  onPickCore?: (coreType: string) => void
  /** Create a fal generation node for a specific model id. */
  onPickGen?: (modelId: string) => void
  onClose: () => void
}): React.JSX.Element {
  // Only high-level model nodes are offered; loaders/samplers/inputs are hidden plumbing.
  const all = addableCoreNodes(coreNodes)
  // Extension nodes get their own section so it's clear which are community-provided.
  const addable = all.filter((n) => !isExtensionNode(n.source))
  const extensionGroups = groupByExtension(all.filter((n) => isExtensionNode(n.source)))
  // Fal models, grouped by owner (OpenAI, ByteDance, …) - listed like the Inline Core section.
  const falGroups = groupByOwner(listNodeDefs())
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
        {/* One scroll area for the whole list (built-ins + fal + Inline Core), not per-section. */}
        <div className="max-h-[70vh] overflow-y-auto">
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
          {onPickGen && falGroups.length > 0 && (
            <div className="border-t border-border">
              <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">
                Fal Models
              </div>
              {falGroups.map((group) => (
                <div key={group.owner}>
                  <div className="px-2.5 pt-1 text-[9px] uppercase tracking-wide text-zinc-600">
                    {group.label}
                  </div>
                  {group.defs.map((def) => (
                    <button
                      key={def.id}
                      onClick={() => onPickGen(def.id)}
                      className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
                    >
                      <span className="flex h-4 w-4 shrink-0 items-center justify-center text-emerald-300">
                        <SparklesIcon />
                      </span>
                      {def.title}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
          {addable.length > 0 && (
            <div className="border-t border-border">
              <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">
                Inline Core
              </div>
              {groupByCategory(addable).map(([category, nodes]) => (
                <div key={category}>
                  <div className="px-2.5 pt-1 text-[9px] uppercase tracking-wide text-zinc-600">
                    {category}
                  </div>
                  {nodes.map((n) => (
                    <button
                      key={n.type}
                      onClick={() => onPickCore?.(n.type)}
                      className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
                    >
                      <span className="flex h-4 w-4 shrink-0 items-center justify-center">
                        <NodeGlyph />
                      </span>
                      {n.title}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
          {extensionGroups.length > 0 && (
            <div className="border-t border-border">
              <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">
                Extensions
              </div>
              {extensionGroups.map(([extension, nodes]) => (
                <div key={extension}>
                  <div className="px-2.5 pt-1 text-[9px] uppercase tracking-wide text-zinc-600">
                    {extension}
                  </div>
                  {nodes.map((n) => (
                    <button
                      key={n.type}
                      onClick={() => onPickCore?.(n.type)}
                      className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
                    >
                      <span className="flex h-4 w-4 shrink-0 items-center justify-center">
                        <NodeGlyph />
                      </span>
                      {n.title}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

/** Extension nodes keyed by their owning extension, from the `ext:<extension>:<module>` source. */
function groupByExtension(nodes: NodeDescriptor[]): Array<[string, NodeDescriptor[]]> {
  const groups = new Map<string, NodeDescriptor[]>()
  for (const node of nodes) {
    const extension = extensionOf(node.source) ?? 'extension'
    const list = groups.get(extension) ?? []
    list.push(node)
    groups.set(extension, list)
  }
  return [...groups.entries()]
}

function groupByCategory(nodes: NodeDescriptor[]): Array<[string, NodeDescriptor[]]> {
  const groups = new Map<string, NodeDescriptor[]>()
  for (const node of nodes) {
    const list = groups.get(node.category) ?? []
    list.push(node)
    groups.set(node.category, list)
  }
  return [...groups.entries()]
}

// ── Node icons (moved here from CanvasToolbar, which no longer shows per-node buttons) ──

function NodeGlyph(): React.JSX.Element {
  return (
    <Svg>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <path d="M4 10h16" />
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
