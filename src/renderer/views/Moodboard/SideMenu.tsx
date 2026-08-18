import { useEffect, useState } from 'react'
import { useFrameStore } from '../../store/frameStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { LibraryPanel } from '../Library/LibraryPanel'
import { OutputThumb, type OutputTile } from '../Library/OutputThumb'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import {
  ChevronLeftIcon,
  CharacterIcon,
  ChevronRightIcon,
  DatasetIcon,
  ImageIcon,
  LoraOutputIcon,
  ModelsIcon,
  SparklesIcon,
} from '../../components/icons'
import { ModelsPanel } from '../Models/ModelsPanel'
import { CharacterLibraryPanel } from '../Characters/CharacterLibraryPanel'
import { DatasetsPanel } from '../Trainer/DatasetsPanel'
import { OutputsPanel } from '../Trainer/OutputsPanel'

type Tab = 'assets' | 'outputs' | 'characters' | 'models' | 'datasets' | 'loras'

const TABS: { key: Tab; label: string; Icon: (p: { className?: string }) => React.JSX.Element }[] =
  [
    { key: 'assets', label: 'Assets', Icon: ImageIcon },
    { key: 'outputs', label: 'Outputs', Icon: SparklesIcon },
    { key: 'characters', label: 'Characters', Icon: CharacterIcon },
    { key: 'models', label: 'Models', Icon: ModelsIcon },
    { key: 'datasets', label: 'Datasets', Icon: DatasetIcon },
    { key: 'loras', label: 'LoRAs', Icon: LoraOutputIcon },
  ]

/** Collapsible left rail for the canvas; node creation lives in the floating toolbar instead. */
const MIN_PANEL_WIDTH = 200
const MAX_PANEL_WIDTH = 600

export function SideMenu(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('assets')
  // Start collapsed to the icon rail: the canvas gets the full width by default.
  const [open, setOpen] = useState(false)
  const [width, setWidth] = useState(256)

  // Drag the right separator to resize the expanded panel. Listeners live on the
  // window so the drag keeps tracking even when the cursor outruns the handle.
  const startResize = (e: React.MouseEvent): void => {
    e.preventDefault()
    const startX = e.clientX
    const startW = width
    const onMove = (ev: MouseEvent): void => {
      const next = Math.min(
        MAX_PANEL_WIDTH,
        Math.max(MIN_PANEL_WIDTH, startW + ev.clientX - startX),
      )
      setWidth(next)
    }
    const onUp = (): void => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  if (!open) {
    return (
      <div className="flex w-11 shrink-0 flex-col items-center gap-1 border-r border-border bg-surface py-2">
        <button
          onClick={() => setOpen(true)}
          title="Expand menu"
          aria-label="Expand menu"
          className="mb-1 flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-panel hover:text-white"
        >
          <ChevronRightIcon className="h-5 w-5" />
        </button>
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => {
              setTab(t.key)
              setOpen(true)
            }}
            title={t.label}
            aria-pressed={tab === t.key}
            className={`flex h-9 w-9 items-center justify-center rounded-md transition-colors ${
              tab === t.key
                ? 'bg-accent text-panel shadow-sm'
                : 'text-zinc-400 hover:bg-panel hover:text-zinc-200'
            }`}
          >
            <t.Icon className="h-5 w-5" />
          </button>
        ))}
      </div>
    )
  }

  return (
    <div
      className="relative flex shrink-0 flex-col border-r border-border bg-surface"
      style={{ width }}
    >
      <div className="flex items-center gap-1 border-b border-border px-1.5 py-1.5">
        {/* Segmented tab control: the active tab shows its label, the rest stay icon-only so
            the row never outgrows the panel and the collapse control is always reachable. */}
        <div className="flex min-w-0 flex-1 items-center gap-0.5 rounded-lg bg-black/20 p-0.5">
          {TABS.map((t) => {
            const active = tab === t.key
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                title={t.label}
                aria-pressed={active}
                className={`flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
                  active
                    ? 'bg-accent text-panel shadow-sm'
                    : 'text-zinc-400 hover:bg-panel hover:text-zinc-200'
                }`}
              >
                <t.Icon className="h-4 w-4 shrink-0" />
                {active && <span className="truncate">{t.label}</span>}
              </button>
            )
          })}
        </div>
        <button
          onClick={() => setOpen(false)}
          title="Collapse menu"
          aria-label="Collapse menu"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-zinc-500 hover:bg-panel hover:text-white"
        >
          <ChevronLeftIcon className="h-4 w-4" />
        </button>
      </div>

      <div className="min-h-0 flex-1">
        {/* Assets reuses the full library panel - drag a tile onto the canvas to create a frame. */}
        {tab === 'assets' && <LibraryPanel />}
        {tab === 'outputs' && <OutputsTab />}
        {tab === 'characters' && <CharacterLibraryPanel />}
        {tab === 'models' && <ModelsPanel />}
        {tab === 'datasets' && <DatasetsPanel />}
        {tab === 'loras' && <OutputsPanel />}
      </div>

      {/* Drag separator on the right edge to resize the panel. */}
      <div
        onMouseDown={startResize}
        title="Drag to resize"
        className="absolute -right-0.5 top-0 z-10 h-full w-1.5 cursor-col-resize hover:bg-accent/40"
      />
    </div>
  )
}

/**
 * Outputs tab - a flat gallery of every generated render, newest first: frame takes (fal)
 * plus Core-node (e.g. Z-Image) outputs, which aren't Frames and live on their canvas item instead.
 * Frame tiles drag onto a generation node to feed it as an input (via the frame's flow link).
 */
function OutputsTab(): React.JSX.Element {
  const frames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)
  const items = useMoodboardStore((s) => s.items)
  const coreDescriptors = useCoreNodesStore((s) => s.descriptors)

  // Ensure frames + their takes and the board's items are loaded even if this tab opens first.
  useEffect(() => {
    void useFrameStore.getState().load()
    void useMoodboardStore.getState().load()
  }, [])

  // Frame takes: draggable (carry their frame id).
  const frameOutputs = frames.flatMap((f) =>
    (takesByFrame[f.id] ?? []).map((take) => ({
      createdAt: take.createdAt,
      onDelete: () => void useFrameStore.getState().deleteTake(take.id),
      tile: {
        id: take.id,
        filePath: take.filePath,
        kind: take.kind,
        label: f.name,
        frameId: take.frameId,
      } satisfies OutputTile,
    })),
  )

  // Core-node renders live on the moodboard item (data.core.outputs), not the takes table, so surface
  // them here too. `outputs` already includes the active `output` and is deduped newest-first server-side.
  const coreOutputs = items.flatMap((it) => {
    const core = it.type === 'core' ? it.data.core : undefined
    if (!core) return []
    const label = coreDescriptors.find((d) => d.type === core.type)?.title ?? core.type
    const history = core.outputs ?? (core.output ? [core.output] : [])
    return history.map((o) => ({
      createdAt: o.createdAt ?? 0,
      onDelete: () => void useMoodboardStore.getState().removeCoreOutput(it.id, o.takeId),
      tile: {
        id: o.takeId,
        filePath: o.filePath,
        kind: o.kind,
        label,
        frameId: null,
      } satisfies OutputTile,
    }))
  })

  const outputs = [...frameOutputs, ...coreOutputs].sort((a, b) => b.createdAt - a.createdAt)

  if (outputs.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1.5 p-6 text-center">
        <SparklesIcon className="h-7 w-7 text-zinc-600" />
        <p className="text-sm text-zinc-500">No outputs yet</p>
        <p className="text-xs text-zinc-600">Generate a frame and its takes show up here.</p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">
          {outputs.length} output{outputs.length === 1 ? '' : 's'}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        <div className="grid grid-cols-2 gap-2">
          {outputs.map(({ tile, onDelete }) => (
            <OutputThumb key={tile.id} tile={tile} onDelete={onDelete} />
          ))}
        </div>
      </div>
    </div>
  )
}
