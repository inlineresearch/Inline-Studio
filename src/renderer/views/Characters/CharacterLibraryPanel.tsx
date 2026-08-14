import { useEffect, useState } from 'react'
import type { CharacterSummary } from '@shared/types'
import { useAssetStore } from '../../store/assetStore'
import { useCharacterStore } from '../../store/characterStore'
import { getAssetDragIds } from '../../lib/dnd'
import { CloseIcon, DownloadIcon, PlusIcon } from '../../components/icons'
import { CharacterEditor } from './CharacterEditor'

/**
 * Left panel: every saved character, plus creating one by dropping images.
 *
 * A character lives in the models root rather than the project, so this list is the same in every
 * project. Creation deliberately accepts a single image: the nudges toward a stronger character
 * are advice on the card afterwards, never a form to fill in first.
 */
export function CharacterLibraryPanel(): React.JSX.Element {
  const { characters, editing, loading, busy, error, load, create, remove, open, importFile } =
    useCharacterStore()
  const loadAssets = useAssetStore((s) => s.load)
  const [draft, setDraft] = useState<{ name: string; assetIds: string[] } | null>(null)
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    void load()
  }, [load])

  const acceptsDrop = (e: React.DragEvent): boolean =>
    Array.from(e.dataTransfer.types).some((t) => t === 'Files' || t.startsWith('application/x-'))

  const onDragOver = (e: React.DragEvent): void => {
    if (!acceptsDrop(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    if (!dragOver) setDragOver(true)
  }

  const onDrop = async (e: React.DragEvent): Promise<void> => {
    if (!acceptsDrop(e)) return
    e.preventDefault()
    setDragOver(false)

    // Assets dragged from the library become a new character's references.
    const assetIds = getAssetDragIds(e.dataTransfer)
    if (assetIds.length > 0) {
      setDraft({ name: '', assetIds })
      return
    }

    const files = Array.from(e.dataTransfer.files ?? [])
    const chars = files.filter((f) => f.name.toLowerCase().endsWith('.char'))
    for (const file of chars) await importFile(file)

    const images = files.filter((f) => f.type.startsWith('image/'))
    if (images.length > 0) {
      // Route new images through the asset library first, so a character's references are also
      // ordinary assets the user can find, re-use and delete.
      await useAssetStore.getState().importFiles(images)
      const added = useAssetStore.getState().assets.slice(0, images.length)
      setDraft({ name: '', assetIds: added.map((a) => a.id) })
    }
  }

  if (editing) return <CharacterEditor />

  return (
    <div
      className={`flex h-full flex-col ${dragOver ? 'ring-2 ring-inset ring-accent' : ''}`}
      onDragOver={onDragOver}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false)
      }}
      onDrop={(e) => void onDrop(e)}
    >
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="text-xs font-medium text-muted">Characters</span>
        <button
          type="button"
          title="Create a character from selected library images"
          onClick={() => {
            void loadAssets()
            setDraft({ name: '', assetIds: [] })
          }}
          className="flex h-6 w-6 items-center justify-center rounded text-muted hover:bg-surface hover:text-fg"
        >
          <PlusIcon className="h-4 w-4" />
        </button>
      </div>

      {error && <div className="px-2 py-1 text-[11px] text-red-400">{error}</div>}
      {busy && <div className="px-2 py-1 text-[11px] text-muted">Building character…</div>}

      {draft && (
        <NewCharacterForm
          draft={draft}
          onChange={setDraft}
          onCancel={() => setDraft(null)}
          onSubmit={async (name, assetIds, description) => {
            const ok = await create(name, assetIds, description)
            if (ok) setDraft(null)
          }}
          busy={busy}
        />
      )}

      <div className="flex-1 space-y-2 overflow-y-auto p-2">
        {characters.map((character) => (
          <CharacterCard
            key={character.file}
            character={character}
            onOpen={() => void open(character.file)}
            onDelete={() => void remove(character.file)}
          />
        ))}
        {!loading && characters.length === 0 && !draft && (
          <div className="rounded border border-dashed border-border p-4 text-center text-[11px] text-muted">
            Drop an image here to make a character. One is enough.
          </div>
        )}
      </div>
    </div>
  )
}

function CharacterCard({
  character,
  onOpen,
  onDelete,
}: {
  character: CharacterSummary
  onOpen: () => void
  onDelete: () => void
}): React.JSX.Element {
  // The X sits under the cursor on hover, and deleting is irreversible, so it asks first.
  const [confirming, setConfirming] = useState(false)
  if (character.error) {
    return (
      <div className="rounded border border-red-500/40 bg-red-500/5 p-2 text-[11px] text-red-300">
        <div className="font-medium">{character.name}</div>
        <div className="mt-0.5 text-red-400/80">{character.error}</div>
      </div>
    )
  }
  return (
    <div className="group relative rounded border border-border bg-surface/60 hover:border-zinc-600">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full items-center gap-2 p-2 text-left"
      >
        <img
          src={`/character-ref/${encodeURIComponent(character.file)}/0`}
          alt=""
          className="h-10 w-10 shrink-0 rounded object-cover"
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs text-fg">{character.name}</span>
          <span className="block truncate text-[10px] text-muted">
            {character.refs} reference{character.refs === 1 ? '' : 's'}
          </span>
        </span>
      </button>
      <div className="absolute right-1 top-1 hidden gap-0.5 group-hover:flex">
        <a
          href={`/download/character/${encodeURIComponent(character.file)}`}
          download={character.file}
          title="Export this character"
          onClick={(e) => e.stopPropagation()}
          className="flex h-5 w-5 items-center justify-center rounded bg-black/60 text-muted hover:text-fg"
        >
          <DownloadIcon className="h-3 w-3" />
        </a>
        <button
          type="button"
          title="Delete this character"
          onClick={(e) => {
            e.stopPropagation()
            setConfirming(true)
          }}
          className="flex h-5 w-5 items-center justify-center rounded bg-black/60 text-muted hover:text-red-400"
        >
          <CloseIcon className="h-3 w-3" />
        </button>
      </div>
      {confirming && (
        <div className="absolute inset-0 flex items-center justify-center gap-1 rounded bg-black/85 px-2">
          <span className="truncate text-[10px] text-muted">Delete?</span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            className="rounded bg-red-600 px-1.5 py-0.5 text-[10px] text-white hover:bg-red-500"
          >
            Delete
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setConfirming(false)
            }}
            className="rounded px-1.5 py-0.5 text-[10px] text-muted hover:text-fg"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

function NewCharacterForm({
  draft,
  onChange,
  onCancel,
  onSubmit,
  busy,
}: {
  draft: { name: string; assetIds: string[] }
  onChange: (draft: { name: string; assetIds: string[] }) => void
  onCancel: () => void
  onSubmit: (name: string, assetIds: string[], description: string) => Promise<void>
  busy: boolean
}): React.JSX.Element {
  const assets = useAssetStore((s) => s.assets)
  const [description, setDescription] = useState('')
  const images = assets.filter((a) => a.kind === 'image')
  const ready = draft.name.trim().length > 0 && draft.assetIds.length > 0

  const toggle = (id: string): void =>
    onChange({
      ...draft,
      assetIds: draft.assetIds.includes(id)
        ? draft.assetIds.filter((x) => x !== id)
        : [...draft.assetIds, id],
    })

  return (
    <div className="space-y-2 border-b border-border p-2">
      <input
        autoFocus
        value={draft.name}
        onChange={(e) => onChange({ ...draft, name: e.target.value })}
        placeholder="Character name"
        className="w-full rounded border border-border bg-black/30 px-2 py-1 text-xs text-fg outline-none focus:border-accent"
      />
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Describe them (optional). This is prepended to every prompt."
        rows={2}
        className="w-full resize-none rounded border border-border bg-black/30 px-2 py-1 text-xs text-fg outline-none focus:border-accent"
      />
      <div className="text-[10px] text-muted">
        {draft.assetIds.length} reference{draft.assetIds.length === 1 ? '' : 's'} selected. Order is
        the order the prompt refers to them.
      </div>
      <div className="grid max-h-40 grid-cols-4 gap-1 overflow-y-auto">
        {images.map((asset) => {
          const position = draft.assetIds.indexOf(asset.id)
          return (
            <button
              key={asset.id}
              type="button"
              onClick={() => toggle(asset.id)}
              className={`relative aspect-square overflow-hidden rounded border ${
                position >= 0 ? 'border-accent' : 'border-border'
              }`}
            >
              <img
                src={`/media/${asset.thumbPath ?? asset.filePath}`}
                alt=""
                className="h-full w-full object-cover"
              />
              {position >= 0 && (
                <span className="absolute left-0.5 top-0.5 rounded bg-accent px-1 text-[9px] text-black">
                  {position + 1}
                </span>
              )}
            </button>
          )
        })}
      </div>
      <div className="flex justify-end gap-1">
        <button
          type="button"
          onClick={onCancel}
          className="rounded px-2 py-1 text-[11px] text-muted hover:text-fg"
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={!ready || busy}
          onClick={() => void onSubmit(draft.name.trim(), draft.assetIds, description)}
          className="rounded bg-emerald-600 px-2 py-1 text-[11px] text-white disabled:opacity-40"
        >
          {busy ? 'Building…' : 'Create'}
        </button>
      </div>
    </div>
  )
}
