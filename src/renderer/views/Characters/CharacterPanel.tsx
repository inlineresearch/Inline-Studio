import { useState } from 'react'
import { CloseIcon, DownloadIcon, TrashIcon } from '../../components/icons'
import { useCharacterStore } from '../../store/characterStore'
import { CharacterCreateForm } from './CharacterCreateForm'
import { CharacterEditor } from './CharacterEditor'
import { EncodeProgress } from './EncodeProgress'

const MIN_WIDTH = 360
/** Wide enough to cover the canvas, but the side panel stays visible: it is the drag source. */
const MAX_WIDTH = 1400

/** Whether this character can be generated with right now, said where the name is. */
function StatusTag({
  busy,
  needsRebuild,
}: {
  busy: boolean
  needsRebuild?: boolean
}): React.JSX.Element {
  const settled = !busy && !needsRebuild
  return (
    <span
      className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] ${
        settled
          ? 'border-emerald-600/40 bg-emerald-500/10 text-emerald-400'
          : 'border-amber-600/40 bg-amber-500/10 text-amber-400'
      }`}
    >
      {busy ? 'Rebuilding' : needsRebuild ? 'Not ready' : 'Ready'}
    </span>
  )
}

/**
 * The character editor as a column beside the side panel rather than a modal, so the Assets and
 * Outputs tabs stay reachable and their thumbnails are dragged in instead of rendered twice.
 */
export function CharacterPanel(): React.JSX.Element | null {
  const panel = useCharacterStore((s) => s.panel)
  const editing = useCharacterStore((s) => s.editing)
  const closePanel = useCharacterStore((s) => s.closePanel)
  const remove = useCharacterStore((s) => s.remove)
  const busy = useCharacterStore((s) => s.busy)
  const [width, setWidth] = useState(560)
  const [confirmDelete, setConfirmDelete] = useState(false)

  // Listeners live on the window so the drag keeps tracking when the cursor outruns the handle.
  const startResize = (e: React.MouseEvent): void => {
    e.preventDefault()
    const startX = e.clientX
    const startW = width
    const onMove = (ev: MouseEvent): void => {
      setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startW + ev.clientX - startX)))
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

  if (!panel) return null
  const creating = panel.kind === 'create'

  return (
    <div
      className="relative flex shrink-0 flex-col border-r border-border bg-surface"
      style={{ width }}
    >
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <h2 className="min-w-0 truncate text-sm font-semibold text-zinc-100">
            {creating ? 'New character' : (editing?.name ?? 'Character')}
          </h2>
          {!creating && editing && <StatusTag busy={busy} needsRebuild={editing.needsRebuild} />}
        </div>
        {!creating && editing && (
          <>
            {confirmDelete ? (
              <span className="flex items-center gap-1">
                <span className="text-[10px] text-muted">Delete?</span>
                <button
                  type="button"
                  onClick={() => void remove(editing.file).then(closePanel)}
                  className="rounded bg-red-600 px-2 py-0.5 text-[10px] text-white hover:bg-red-500"
                >
                  Delete
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="rounded px-1.5 py-0.5 text-[10px] text-muted hover:text-fg"
                >
                  Cancel
                </button>
              </span>
            ) : (
              <button
                type="button"
                title="Delete this character"
                onClick={() => setConfirmDelete(true)}
                className="rounded p-1 text-zinc-400 hover:text-red-400"
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            )}
            <a
              href={`/download/character/${encodeURIComponent(editing.file)}`}
              download={editing.file}
              title="Export this character"
              className="rounded p-1 text-zinc-400 hover:text-zinc-100"
            >
              <DownloadIcon className="h-4 w-4" />
            </a>
          </>
        )}
        <button
          type="button"
          title="Close"
          onClick={closePanel}
          className="rounded p-1 text-zinc-400 hover:text-zinc-100"
        >
          <CloseIcon className="h-4 w-4" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {creating ? (
          <CharacterCreateForm
            initialAssetIds={panel.assetIds}
            onDone={closePanel}
            onCancel={closePanel}
          />
        ) : editing ? (
          <CharacterEditor />
        ) : (
          // Opening a character written before the current encoders rebuilds its scoring first.
          <div className="p-5">
            <EncodeProgress label="Opening…" />
          </div>
        )}
      </div>

      <div
        onMouseDown={startResize}
        title="Drag to resize"
        className="absolute -right-0.5 top-0 z-10 h-full w-1.5 cursor-col-resize hover:bg-accent/40"
      />
    </div>
  )
}
