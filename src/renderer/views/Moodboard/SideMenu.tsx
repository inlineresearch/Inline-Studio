import { useEffect, useState } from 'react'
import { takeWaveformPath } from '@shared/media'
import type { Frame } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { useFrameStore } from '../../store/frameStore'
import { useAssetStore } from '../../store/assetStore'
import { useMoodboardStore } from '../../store/moodboardStore'
import { useUiStore } from '../../store/uiStore'
import { LibraryPanel } from '../Library/LibraryPanel'
import { OutputThumb } from '../Library/OutputThumb'
import { setFrameDragPayload } from '../../lib/dnd'
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
  WorkflowIcon,
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
 * each frame as a folder of Inputs / Outputs / Workflow, with delete + sort. Node
 * creation lives in the floating canvas toolbar instead.
 */
const MIN_PANEL_WIDTH = 200
const MAX_PANEL_WIDTH = 600

export function SideMenu(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('assets')
  const [open, setOpen] = useState(true)
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
        {/* Assets reuses the full library panel — drag a tile onto the canvas to create a frame. */}
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
 * Outputs tab — a flat gallery of every generated take across all frames, newest first.
 * Each tile drags onto a generation node to feed it as an input (via its frame's flow link).
 */
function OutputsTab(): React.JSX.Element {
  const frames = useFrameStore((s) => s.frames)
  const takesByFrame = useFrameStore((s) => s.takesByFrame)

  // Ensure frames + their takes are loaded even when the user opens this tab first.
  useEffect(() => {
    void useFrameStore.getState().load()
  }, [])

  const outputs = frames
    .flatMap((f) => (takesByFrame[f.id] ?? []).map((take) => ({ take, frameName: f.name })))
    .sort((a, b) => b.take.createdAt - a.take.createdAt)

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
          {outputs.map(({ take, frameName }) => (
            <OutputThumb key={take.id} take={take} frameName={frameName} />
          ))}
        </div>
      </div>
    </div>
  )
}

function TimelineTab(): React.JSX.Element {
  const frames = useFrameStore((s) => s.frames)
  const removeFrame = useFrameStore((s) => s.remove)
  const reloadBoard = useMoodboardStore((s) => s.load)
  const [sort, setSort] = useState<SortKey>('updated')

  const sorted = [...frames].sort((a, b) =>
    sort === 'name'
      ? a.name.localeCompare(b.name, undefined, { numeric: true })
      : b.updatedAt - a.updatedAt,
  )

  const onDelete = async (frame: Frame): Promise<void> => {
    if (
      !window.confirm(
        `Delete Frame ${frame.name}? Its takes, workflow and canvas node are removed.`,
      )
    )
      return
    await removeFrame(frame.id)
    void reloadBoard() // drop the (now-deleted) canvas node
  }

  if (frames.length === 0) {
    return (
      <p className="p-2 text-xs text-zinc-600">No frames yet — drag an asset onto the canvas.</p>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">
          {frames.length} frame{frames.length === 1 ? '' : 's'}
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
          {sorted.map((frame) => (
            <FrameFolder key={frame.id} frame={frame} onDelete={() => void onDelete(frame)} />
          ))}
        </div>
      </div>
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
  const workflowFile = frame.comfyWorkflowName
    ? `${frame.comfyWorkflowName.split('/').pop()}.json`
    : null

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
        {frame.comfyWorkflowName && (
          <span title="Linked workflow" className="flex shrink-0 text-zinc-400">
            <WorkflowIcon className="h-3.5 w-3.5" />
          </span>
        )}
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

          <Folder label="Workflow" count={workflowFile ? 1 : 0}>
            {workflowFile ? (
              <div className="flex items-center gap-1 py-0.5 text-[11px] text-zinc-300">
                <span className="text-zinc-500">{'{ }'}</span>
                <span className="min-w-0 flex-1 truncate" title={frame.comfyWorkflowName ?? ''}>
                  {workflowFile}
                </span>
                <span className="text-[9px] text-zinc-600">saved</span>
              </div>
            ) : (
              <Empty>open the frame to create it</Empty>
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
