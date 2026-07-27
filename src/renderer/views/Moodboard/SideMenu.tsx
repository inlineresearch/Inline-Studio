import { useEffect, useState } from 'react'
import { takeWaveformPath } from '@shared/media'
import type { Frame, MoodboardItem } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { useFrameStore } from '../../store/frameStore'
import { useAssetStore } from '../../store/assetStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useUiStore } from '../../store/uiStore'
import { LibraryPanel } from '../Library/LibraryPanel'
import { OutputThumb, type OutputTile } from '../Library/OutputThumb'
import { useCoreNodesStore } from '../../store/coreNodesStore'
import { setFrameDragPayload, setMediaFileDragPayload } from '../../lib/dnd'
import {
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  EditIcon,
  FolderIcon,
  HistoryIcon,
  ImageIcon,
  MusicNoteIcon,
  SparklesIcon,
  StarIcon,
} from '../../components/icons'
import { Waveform } from '../../components/Waveform'

type Tab = 'assets' | 'outputs' | 'timeline'
type SortKey = 'updated' | 'name'

const TABS: { key: Tab; label: string; Icon: (p: { className?: string }) => React.JSX.Element }[] =
  [
    { key: 'assets', label: 'Assets', Icon: ImageIcon },
    { key: 'outputs', label: 'Outputs', Icon: SparklesIcon },
    { key: 'timeline', label: 'Timeline', Icon: HistoryIcon },
  ]

/**
 * Collapsible left rail for the canvas. Assets reuses the full library (browse /
 * import / folders; drag a tile onto the canvas to create a frame). Timeline shows
 * each frame as a folder of Inputs / Outputs, with delete + sort. Node
 * creation lives in the floating canvas toolbar instead.
 */
const MIN_PANEL_WIDTH = 200
const MAX_PANEL_WIDTH = 600

export function SideMenu(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('assets')
  // Start collapsed to the icon rail: the canvas gets the full width by default, and the user
  // expands Assets/Outputs/Timeline when they want them.
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
        {tab === 'timeline' && <TimelineTab />}
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

/** One generation node in the Timeline: a fal frame or a Core-node moodboard item. */
type TimelineNode =
  | { kind: 'frame'; key: string; label: string; updatedAt: number; frame: Frame }
  | { kind: 'core'; key: string; label: string; updatedAt: number; item: MoodboardItem }

function TimelineTab(): React.JSX.Element {
  const frames = useFrameStore((s) => s.frames)
  const removeFrame = useFrameStore((s) => s.remove)
  const items = useMoodboardStore((s) => s.items)
  const deleteItem = useMoodboardStore((s) => s.deleteItem)
  const reloadBoard = useMoodboardStore((s) => s.load)
  const coreDescriptors = useCoreNodesStore((s) => s.descriptors)
  const [sort, setSort] = useState<SortKey>('updated')

  // Both fal frames and Core nodes are generation nodes - surface them together.
  const nodes: TimelineNode[] = [
    ...frames.map(
      (frame): TimelineNode => ({
        kind: 'frame',
        key: frame.id,
        label: `Frame ${frame.name}`,
        updatedAt: frame.updatedAt,
        frame,
      }),
    ),
    ...items
      .filter((it) => it.type === 'core' && it.data.core)
      .map((item): TimelineNode => {
        const core = item.data.core!
        return {
          kind: 'core',
          key: item.id,
          label: coreDescriptors.find((d) => d.type === core.type)?.title ?? core.type,
          updatedAt: item.updatedAt,
          item,
        }
      }),
  ]

  const sorted = [...nodes].sort((a, b) =>
    sort === 'name'
      ? a.label.localeCompare(b.label, undefined, { numeric: true })
      : b.updatedAt - a.updatedAt,
  )

  const onDeleteFrame = async (frame: Frame): Promise<void> => {
    if (!window.confirm(`Delete Frame ${frame.name}? Its takes and canvas node are removed.`))
      return
    await removeFrame(frame.id)
    void reloadBoard() // drop the (now-deleted) canvas node
  }

  const onDeleteCore = async (item: MoodboardItem, label: string): Promise<void> => {
    if (!window.confirm(`Delete ${label}? Its canvas node and renders are removed.`)) return
    await deleteItem(item.id)
  }

  if (sorted.length === 0) {
    return (
      <p className="p-2 text-xs text-zinc-600">
        No generation nodes yet - add a fal or Core node to the canvas.
      </p>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">
          {sorted.length} node{sorted.length === 1 ? '' : 's'}
        </span>
        <label className="flex items-center gap-1 text-[10px] text-zinc-500">
          Sort
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            className="rounded border border-border bg-surface px-1 py-0.5 text-[10px] text-zinc-300 outline-none"
          >
            <option value="updated">Last updated</option>
            <option value="name">Name A–Z</option>
          </select>
        </label>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        <div className="flex flex-col gap-1">
          {sorted.map((n) =>
            n.kind === 'frame' ? (
              <FrameFolder
                key={n.key}
                frame={n.frame}
                onDelete={() => void onDeleteFrame(n.frame)}
              />
            ) : (
              <CoreFolder
                key={n.key}
                item={n.item}
                label={n.label}
                onDelete={() => void onDeleteCore(n.item, n.label)}
              />
            ),
          )}
        </div>
      </div>
    </div>
  )
}

/** A Core-node row in the Timeline: title + its renders (draggable onto the canvas). */
function CoreFolder({
  item,
  label,
  onDelete,
}: {
  item: MoodboardItem
  label: string
  onDelete: () => void
}): React.JSX.Element {
  const [open, setOpen] = useState(false)
  const core = item.data.core
  const outputs = core?.outputs ?? (core?.output ? [core.output] : [])

  return (
    <div className="overflow-hidden rounded border border-border">
      <div className="flex items-center gap-1 bg-surface px-1.5 py-1">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex min-w-0 flex-1 items-center gap-1 text-left"
          title="Toggle"
        >
          {open ? (
            <ChevronDownIcon className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
          ) : (
            <ChevronRightIcon className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
          )}
          <SparklesIcon className="h-3 w-3 shrink-0 text-emerald-400" />
          <span className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-200">{label}</span>
        </button>
        <button
          onClick={onDelete}
          title="Delete node"
          className="flex items-center px-1 text-zinc-400 hover:text-red-400"
        >
          <CloseIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {open && (
        <div className="border-t border-border py-1 pl-2 pr-1.5">
          <Folder label="Outputs" count={outputs.length}>
            {outputs.length === 0 ? (
              <Empty>none</Empty>
            ) : (
              outputs.map((o) => (
                <div
                  key={o.takeId}
                  draggable
                  onDragStart={(e) =>
                    setMediaFileDragPayload(e.dataTransfer, {
                      filePath: o.filePath,
                      kind: o.kind,
                      name: label,
                    })
                  }
                  title="Drag onto the canvas to place this render"
                  className="cursor-grab active:cursor-grabbing"
                >
                  <FileRow
                    name={o.filePath.split('/').pop() ?? 'render'}
                    thumb={resolveMedia(o.filePath)}
                    kind={o.kind}
                    hero={o.takeId === core?.output?.takeId}
                  />
                </div>
              ))
            )}
          </Folder>
        </div>
      )}
    </div>
  )
}

function FrameFolder({
  frame,
  onDelete,
}: {
  frame: Frame
  onDelete: () => void
}): React.JSX.Element {
  const inputs = useFrameStore((s) => s.inputsByFrame[frame.id]) ?? []
  const takes = useFrameStore((s) => s.takesByFrame[frame.id]) ?? []
  const assets = useAssetStore((s) => s.assets)
  const openInspector = useUiStore((s) => s.setInspectorFrame)
  const [open, setOpen] = useState(false)

  const inputAssets = inputs
    .map((i) => assets.find((a) => a.id === i.assetId))
    .filter((a): a is NonNullable<typeof a> => !!a)

  return (
    <div className="overflow-hidden rounded border border-border">
      <div
        draggable
        onDragStart={(e) => setFrameDragPayload(e.dataTransfer, frame.id)}
        title="Drag onto the canvas to place this frame"
        className="flex cursor-grab items-center gap-1 bg-surface px-1.5 py-1 active:cursor-grabbing"
      >
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex min-w-0 flex-1 items-center gap-1 text-left"
          title="Toggle"
        >
          {open ? (
            <ChevronDownIcon className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
          ) : (
            <ChevronRightIcon className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
          )}
          <FolderIcon className="h-3 w-3 shrink-0 text-zinc-500" />
          <span className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-200">
            Frame {frame.name}
          </span>
        </button>
        <button
          onClick={() => openInspector(frame.id)}
          title="Edit frame"
          className="px-1 text-zinc-400 hover:text-white"
        >
          <EditIcon className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={onDelete}
          title="Delete frame"
          className="flex items-center px-1 text-zinc-400 hover:text-red-400"
        >
          <CloseIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      {open && (
        <div className="border-t border-border py-1 pl-2 pr-1.5">
          <Folder label="Inputs" count={inputAssets.length}>
            {inputAssets.length === 0 ? (
              <Empty>none</Empty>
            ) : (
              inputAssets.map((a) => (
                <FileRow
                  key={a.id}
                  name={a.name}
                  thumb={resolveMedia(a.previewPath ?? a.filePath)}
                  kind={a.kind}
                  poster={a.kind === 'video' && a.thumbPath ? resolveMedia(a.thumbPath) : undefined}
                  waveform={
                    a.kind === 'audio' && a.thumbPath ? resolveMedia(a.thumbPath) : undefined
                  }
                />
              ))
            )}
          </Folder>

          <Folder label="Outputs" count={takes.length}>
            {takes.length === 0 ? (
              <Empty>none</Empty>
            ) : (
              takes.map((t) => (
                <FileRow
                  key={t.id}
                  name={t.filePath.split('/').pop() ?? 'take'}
                  thumb={resolveMedia(t.filePath)}
                  kind={t.kind}
                  hero={t.id === frame.heroTakeId}
                  waveform={t.kind === 'audio' ? resolveMedia(takeWaveformPath(t.id)) : undefined}
                />
              ))
            )}
          </Folder>
        </div>
      )}
    </div>
  )
}

function Folder({
  label,
  count,
  children,
}: {
  label: string
  count: number
  children: React.ReactNode
}): React.JSX.Element {
  const [open, setOpen] = useState(true)
  return (
    <div className="flex flex-col">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 py-0.5 text-left text-[10px] uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
      >
        {open ? (
          <ChevronDownIcon className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRightIcon className="h-3 w-3 shrink-0" />
        )}
        <FolderIcon className="h-3 w-3 shrink-0 text-zinc-500" />
        {label}
        <span className="text-zinc-600">({count})</span>
      </button>
      {open && <div className="flex flex-col gap-0.5 pb-1 pl-3">{children}</div>}
    </div>
  )
}

function FileRow({
  name,
  thumb,
  kind,
  hero,
  poster,
  waveform,
}: {
  name: string
  thumb: string
  kind: 'image' | 'video' | 'audio'
  /** Marks the frame's chosen hero take with a star. */
  hero?: boolean
  poster?: string
  waveform?: string
}): React.JSX.Element {
  return (
    <div className="flex items-center gap-1.5 py-0.5">
      <div className="h-7 w-7 shrink-0 overflow-hidden rounded border border-border bg-black/40">
        {kind === 'image' && <img src={thumb} alt="" className="h-full w-full object-cover" />}
        {kind === 'video' && (
          <video
            src={thumb}
            poster={poster}
            muted
            preload="metadata"
            className="h-full w-full object-cover"
          />
        )}
        {kind === 'audio' &&
          (waveform ? (
            <Waveform url={waveform} bars={24} className="h-full w-full p-0.5 text-emerald-400" />
          ) : (
            <span className="flex h-full w-full items-center justify-center text-zinc-500">
              <MusicNoteIcon className="h-4 w-4" />
            </span>
          ))}
      </div>
      <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-400" title={name}>
        {name}
      </span>
      {hero && (
        <span title="Hero take" className="shrink-0 text-amber-300">
          <StarIcon className="h-3 w-3" />
        </span>
      )}
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }): React.JSX.Element {
  return <span className="py-0.5 text-[10px] text-zinc-600">{children}</span>
}
