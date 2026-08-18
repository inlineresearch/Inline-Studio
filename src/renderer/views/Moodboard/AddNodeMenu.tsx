/**
 * The "Add node" popup opened from the toolbar's + button, or by double-clicking empty canvas in
 * Select mode.
 *
 * Two tabs, because one flat list outgrew the popup: **Core** is everything that runs locally (the
 * canvas built-ins, the Inline Core engine nodes, and extension nodes), **API** is the hosted fal
 * models. Loaders come first in Core - they are the plumbing you reach for while wiring a graph, so
 * they should not sit at the bottom past every model.
 *
 * Positioning mirrors MoodboardPanel's "Connect to…" menu (container-relative left/top).
 */

import { useState } from 'react'

import { useMenuPlacement } from './useMenuPlacement'

import { addableCoreNodes, type NodeDescriptor } from '@shared/coreNodes'
import { isExtensionNode, extensionOf } from '@shared/extensions'
import { listNodeDefs, groupByOwner } from '@shared/nodes/registry'
import { CaptionGlyph, ChartIcon, CpuIcon, LayersIcon, WandIcon } from './nodes/NodeBadge'

/** The node kinds the Add menu can create (Text has its own toolbar tool, so it's not here). */
export type AddNodeKind =
  | 'load'
  | 'layer'
  | 'preview'
  | 'director'
  | 'trim'
  | 'prompt'
  | 'controlSpace'
  | 'train/dataset'
  | 'train/caption'
  | 'train/lora'
  | 'train/loss'
  | 'resource'

type Tab = 'core' | 'api'

/** The category Core uses for its `load/*` nodes; Load Assets joins them under one header. */
const LOADERS = 'Loaders'
const CANVAS = 'Canvas'
const TRAINING = 'Training'

/**
 * Section order in the Core tab. Core serves its categories in registration order, which puts
 * Generate last behind the decomposed primitives - the opposite of how often each is reached for.
 * Anything not listed keeps its served order, after these.
 */
const CORE_ORDER = [LOADERS, 'Generate', CANVAS]

interface Entry {
  kind: AddNodeKind
  label: string
  icon: React.JSX.Element
  category: string
}

const ENTRIES: Entry[] = [
  { kind: 'load', label: 'Load Assets', icon: <ImageIcon />, category: LOADERS },
  { kind: 'layer', label: 'Layer', icon: <LayerIcon />, category: CANVAS },
  { kind: 'preview', label: 'Preview', icon: <ImageIcon />, category: CANVAS },
  { kind: 'director', label: 'Video Director', icon: <ClapperboardIcon />, category: CANVAS },
  { kind: 'trim', label: 'Edit Video/Audio', icon: <ScissorsIcon />, category: CANVAS },
  { kind: 'prompt', label: 'Prompt', icon: <PromptIcon />, category: CANVAS },
  { kind: 'controlSpace', label: 'Control Space', icon: <PoseIcon />, category: CANVAS },
  // Training sorts last because it is not in CORE_ORDER, which is where it belongs: reached for
  // far less often than generating.
  { kind: 'train/dataset', label: 'Load Dataset', icon: <LayersIcon />, category: TRAINING },
  { kind: 'train/caption', label: 'Caption', icon: <CaptionGlyph />, category: TRAINING },
  { kind: 'train/lora', label: 'Train LoRA', icon: <WandIcon />, category: TRAINING },
  { kind: 'train/loss', label: 'Graph', icon: <ChartIcon />, category: TRAINING },
  { kind: 'resource', label: 'Resources', icon: <CpuIcon />, category: TRAINING },
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
  const [tab, setTab] = useState<Tab>('core')
  const place = useMenuPlacement<HTMLDivElement>(x, y, above)

  // Only high-level model nodes are offered; samplers/inputs are hidden plumbing.
  const all = addableCoreNodes(coreNodes)
  const builtIn = all.filter((n) => !isExtensionNode(n.source))
  // Extension nodes get their own section so it's clear which are community-provided.
  const extensionGroups = groupByExtension(all.filter((n) => isExtensionNode(n.source)))
  // Canvas built-ins first within a section, then Core's own nodes - so Loaders reads
  // "Load Assets, Load Diffusion Model, Load VAE, …" under one header.
  const sections = orderSections([
    ...groupEntries().map(([category, entries]): Section => [category, entryRows(entries, onPick)]),
    ...groupByCategory(builtIn).map(
      ([category, nodes]): Section => [category, coreRows(nodes, onPickCore)],
    ),
  ])
  // Fal models, grouped by owner (OpenAI, ByteDance, …).
  const falGroups = onPickGen ? groupByOwner(listNodeDefs()) : []

  return (
    <>
      <div className="absolute inset-0 z-20" onClick={onClose} />
      <div
        ref={place.ref}
        className={`absolute z-30 flex w-64 select-none flex-col overflow-hidden rounded-md border border-border bg-panel text-xs shadow-xl ${
          above ? '-translate-x-1/2 -translate-y-full' : place.flipped ? '-translate-y-full' : ''
        }`}
        style={place.style}
      >
        <div className="flex border-b border-border">
          <TabButton active={tab === 'core'} onClick={() => setTab('core')}>
            Core
          </TabButton>
          <TabButton active={tab === 'api'} onClick={() => setTab('api')}>
            API
          </TabButton>
        </div>

        {/* One scroll area for the active tab, not per-section. `nowheel` keeps React Flow from
            swallowing the wheel and zooming the canvas instead of scrolling this list. */}
        <div className="nowheel overflow-y-auto" style={{ maxHeight: place.maxHeight }}>
          {tab === 'core' ? (
            <>
              {sections.map(([category, rows]) => (
                <Group key={category} label={category}>
                  {rows}
                </Group>
              ))}

              {extensionGroups.length > 0 && (
                <Group label="Extensions">
                  {extensionGroups.map(([extension, nodes]) => (
                    <div key={extension}>
                      <SubHeader>{extension}</SubHeader>
                      {nodes.map((n) => (
                        <Row key={n.type} icon={<NodeGlyph />} onClick={() => onPickCore?.(n.type)}>
                          {n.title}
                        </Row>
                      ))}
                    </div>
                  ))}
                </Group>
              )}
            </>
          ) : falGroups.length > 0 ? (
            falGroups.map((group) => (
              <Group key={group.owner} label={group.label}>
                {group.defs.map((def) => (
                  <Row
                    key={def.id}
                    icon={<SparklesIcon />}
                    accent
                    onClick={() => onPickGen?.(def.id)}
                  >
                    {def.title}
                  </Row>
                ))}
              </Group>
            ))
          ) : (
            <div className="px-2.5 py-3 text-[11px] text-zinc-500">No hosted models available.</div>
          )}
        </div>
      </div>
    </>
  )
}

/** A Core tab section: its header and its rows. */
type Section = [string, React.JSX.Element[]]

function groupEntries(): Array<[string, Entry[]]> {
  const groups = new Map<string, Entry[]>()
  for (const entry of ENTRIES) {
    const list = groups.get(entry.category) ?? []
    list.push(entry)
    groups.set(entry.category, list)
  }
  return [...groups.entries()]
}

function entryRows(entries: Entry[], onPick: (kind: AddNodeKind) => void): React.JSX.Element[] {
  return entries.map((e) => (
    <Row key={e.kind} icon={e.icon} onClick={() => onPick(e.kind)}>
      {e.label}
    </Row>
  ))
}

function coreRows(
  nodes: NodeDescriptor[],
  onPickCore?: (coreType: string) => void,
): React.JSX.Element[] {
  return nodes.map((n) => (
    <Row key={n.type} icon={<NodeGlyph />} onClick={() => onPickCore?.(n.type)}>
      {n.title}
    </Row>
  ))
}

/** Merge same-named sections, then sort by CORE_ORDER; unlisted keep their served order. */
function orderSections(sections: Section[]): Section[] {
  const merged = new Map<string, React.JSX.Element[]>()
  for (const [category, rows] of sections) {
    merged.set(category, [...(merged.get(category) ?? []), ...rows])
  }
  const rank = (category: string): number => {
    const index = CORE_ORDER.indexOf(category)
    return index === -1 ? CORE_ORDER.length : index
  }
  return [...merged.entries()]
    .filter(([, rows]) => rows.length > 0)
    .sort(([a], [b]) => rank(a) - rank(b))
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <button
      onClick={onClick}
      className={`flex-1 px-2.5 py-1.5 text-[11px] font-medium ${
        active
          ? 'border-b border-zinc-300 text-zinc-100'
          : 'border-b border-transparent text-zinc-500 hover:text-zinc-300'
      }`}
    >
      {children}
    </button>
  )
}

function Group({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <div className="border-t border-border first:border-t-0">
      <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      {children}
    </div>
  )
}

function SubHeader({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <div className="px-2.5 pt-1 text-[9px] uppercase tracking-wide text-zinc-600">{children}</div>
  )
}

function Row({
  icon,
  accent,
  onClick,
  children,
}: {
  icon: React.JSX.Element
  /** Accent the glyph (a hosted model is the AI action). */
  accent?: boolean
  onClick: () => void
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <button
      onClick={onClick}
      className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-zinc-200 hover:bg-surface"
    >
      <span
        className={`flex h-4 w-4 shrink-0 items-center justify-center ${
          accent ? 'text-emerald-300' : ''
        }`}
      >
        {icon}
      </span>
      {children}
    </button>
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

function PoseIcon(): React.JSX.Element {
  return (
    <Svg>
      <circle cx="12" cy="5" r="2" />
      <path d="M12 7v6" />
      <path d="M8 9h8" />
      <path d="m12 13-3 6" />
      <path d="m12 13 3 6" />
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
