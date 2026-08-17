import { useState } from 'react'
import { useAssetStore } from '../../store/assetStore'
import { useCharacterStore } from '../../store/characterStore'
import { getAssetDragIds, getMediaFileDrag } from '../../lib/dnd'
import { CloseIcon } from '../../components/icons'
import { EncodeProgress } from './EncodeProgress'

/** Creating a character: references come from the Assets and Outputs tabs by drag, never a second copy of the library. */
export function CharacterCreateForm({
  initialAssetIds,
  onDone,
  onCancel,
}: {
  initialAssetIds: string[]
  onDone: () => void
  onCancel: () => void
}): React.JSX.Element {
  const busy = useCharacterStore((s) => s.busy)
  const create = useCharacterStore((s) => s.create)
  const assets = useAssetStore((s) => s.assets)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [assetIds, setAssetIds] = useState(initialAssetIds)
  const [dragOver, setDragOver] = useState(false)

  const ready = name.trim().length > 0 && assetIds.length > 0
  const chosen = assetIds
    .map((id) => assets.find((a) => a.id === id))
    .filter((a): a is NonNullable<typeof a> => Boolean(a))

  const add = (ids: string[]): void =>
    setAssetIds((current) => [...current, ...ids.filter((id) => !current.includes(id))])

  const onDrop = async (e: React.DragEvent): Promise<void> => {
    e.preventDefault()
    setDragOver(false)
    const dragged = getAssetDragIds(e.dataTransfer)
    if (dragged.length > 0) return add(dragged)

    // An output or an OS file is not an asset yet, so it joins the library before it can be a ref.
    const media = getMediaFileDrag(e.dataTransfer)
    const files = media
      ? [
          await fetch(`/media/${media.filePath}`).then(
            async (r) => new File([await r.blob()], media.name || 'render.png'),
          ),
        ]
      : Array.from(e.dataTransfer.files ?? []).filter((f) => f.type.startsWith('image/'))
    if (files.length === 0) return
    await useAssetStore.getState().importFiles(files)
    add(
      useAssetStore
        .getState()
        .assets.slice(0, files.length)
        .map((a) => a.id),
    )
  }

  return (
    <div
      className={`space-y-4 p-5 ${dragOver ? 'ring-2 ring-inset ring-accent' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false)
      }}
      onDrop={(e) => void onDrop(e)}
    >
      {busy && <EncodeProgress label="Building character…" />}

      <label className="block space-y-1">
        <span className="text-[10px] uppercase tracking-wide text-muted">Name</span>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Character name"
          className="w-full rounded border border-border bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent"
        />
      </label>

      <label className="block space-y-1">
        <span className="text-[10px] uppercase tracking-wide text-muted">Describe character</span>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
          placeholder="Describe them. Prepended to every prompt, and what a trained adapter binds to."
          className="w-full resize-none rounded border border-border bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent"
        />
      </label>

      <div className="space-y-1.5">
        <span className="text-[10px] uppercase tracking-wide text-muted">
          References ({chosen.length})
        </span>
        {chosen.length > 0 && (
          <div className="grid grid-cols-6 gap-1.5">
            {chosen.map((asset, index) => (
              <div
                key={asset.id}
                className="group relative aspect-square overflow-hidden rounded border border-border"
              >
                <img
                  src={`/media/${asset.thumbPath ?? asset.filePath}`}
                  alt=""
                  className="h-full w-full object-cover"
                />
                <span className="absolute left-0.5 top-0.5 rounded bg-black/70 px-1 text-[9px] text-fg">
                  {index + 1}
                </span>
                <button
                  type="button"
                  title="Remove this reference"
                  onClick={() => setAssetIds((ids) => ids.filter((id) => id !== asset.id))}
                  className="absolute right-0.5 top-0.5 hidden h-4 w-4 items-center justify-center rounded bg-black/70 text-muted group-hover:flex hover:text-red-400"
                >
                  <CloseIcon className="h-2.5 w-2.5" />
                </button>
              </div>
            ))}
          </div>
        )}
        <p className="rounded border border-dashed border-border px-2 py-1.5 text-[10px] text-muted">
          Drag and drop images from your computer, Assets or Outputs. One is enough.
        </p>
      </div>

      <div className="flex justify-end gap-2 border-t border-border pt-3">
        <button
          type="button"
          onClick={onCancel}
          className="rounded px-3 py-1.5 text-[11px] text-muted hover:text-fg"
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={!ready || busy}
          onClick={() =>
            void create(name.trim(), assetIds, description).then((ok) => {
              if (ok) onDone()
            })
          }
          className="rounded bg-emerald-600 px-3 py-1.5 text-[11px] text-white hover:bg-emerald-500 disabled:opacity-40"
        >
          {busy ? 'Building…' : 'Create character'}
        </button>
      </div>
    </div>
  )
}
