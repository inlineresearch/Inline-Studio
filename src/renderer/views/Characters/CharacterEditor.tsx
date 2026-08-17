import { useEffect, useState } from 'react'
import { useAssetStore } from '../../store/assetStore'
import { useCharacterStore } from '../../store/characterStore'
import { getAssetDragIds } from '../../lib/dnd'
import { CloseIcon, PlusIcon } from '../../components/icons'
import { pickFilesViaInput } from '../../lib/importFiles'
import { EncodeProgress } from './EncodeProgress'

/**
 * One character open for editing: its name, its locked description, and its references.
 *
 * Editing the description is cheap; adding or removing a reference recompiles the character, so
 * those show the busy state. The reference numbers are the numbers the prompt addresses, which is
 * why order is shown rather than left implicit.
 */
export function CharacterEditor(): React.JSX.Element {
  const { editing, busy, error, closeEditor, rename, setDescription, addRefs, removeRef, remove } =
    useCharacterStore()
  const [name, setName] = useState(editing?.name ?? '')
  const [description, setDescriptionDraft] = useState(editing?.description ?? '')
  const [dragOver, setDragOver] = useState(false)
  // Deleting unlinks the .char, taking its references and description with it, so it asks first.
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => {
    setName(editing?.name ?? '')
    setDescriptionDraft(editing?.description ?? '')
  }, [editing?.file, editing?.name, editing?.description])

  if (!editing) return <div />

  const file = editing.file
  const nameDirty = name.trim() !== editing.name && name.trim().length > 0
  const descriptionDirty = description !== (editing.description ?? '')

  // Images arrive from a drop or from a hint; both land in the library first, so a character's
  // references stay ordinary assets the user can find and reuse.
  const addImages = async (images: File[]): Promise<void> => {
    if (images.length === 0) return
    await useAssetStore.getState().importFiles(images)
    const added = useAssetStore.getState().assets.slice(0, images.length)
    await addRefs(
      file,
      added.map((a) => a.id),
    )
  }

  const onDrop = async (e: React.DragEvent): Promise<void> => {
    e.preventDefault()
    setDragOver(false)
    const assetIds = getAssetDragIds(e.dataTransfer)
    if (assetIds.length > 0) return void addRefs(file, assetIds)
    await addImages(
      Array.from(e.dataTransfer.files ?? []).filter((f) => f.type.startsWith('image/')),
    )
  }

  return (
    <div
      className={`flex h-full flex-col ${dragOver ? 'ring-2 ring-inset ring-accent' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false)
      }}
      onDrop={(e) => void onDrop(e)}
    >
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <button type="button" onClick={closeEditor} className="text-xs text-muted hover:text-fg">
          ← Characters
        </button>
      </div>

      {busy && <EncodeProgress label="Rebuilding…" />}

      {error && <div className="px-2 py-1 text-[11px] text-red-400">{error}</div>}

      <div className="flex-1 space-y-3 overflow-y-auto p-2">
        <label className="block space-y-1">
          <span className="text-[10px] uppercase tracking-wide text-muted">Name</span>
          <span className="flex gap-1">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && nameDirty) void rename(file, name.trim())
              }}
              className="min-w-0 flex-1 rounded border border-border bg-black/30 px-2 py-1 text-xs text-fg outline-none focus:border-accent"
            />
            {nameDirty && (
              <button
                type="button"
                onClick={() => void rename(file, name.trim())}
                className="rounded bg-emerald-600 px-2 text-[11px] text-white"
              >
                Save
              </button>
            )}
          </span>
        </label>

        <label className="block space-y-1">
          <span className="text-[10px] uppercase tracking-wide text-muted">Description</span>
          <textarea
            value={description}
            onChange={(e) => setDescriptionDraft(e.target.value)}
            rows={3}
            placeholder="Prepended to every prompt that uses this character."
            className="w-full resize-none rounded border border-border bg-black/30 px-2 py-1 text-xs text-fg outline-none focus:border-accent"
          />
          {descriptionDirty && (
            <button
              type="button"
              onClick={() => void setDescription(file, description)}
              className="rounded bg-emerald-600 px-2 py-1 text-[11px] text-white"
            >
              Update description
            </button>
          )}
        </label>

        <div className="space-y-1">
          <span className="text-[10px] uppercase tracking-wide text-muted">
            References ({editing.refs})
          </span>
          <div className="grid grid-cols-3 gap-1">
            {editing.refUrls.map((url, index) => {
              const flagged = (editing.flaggedRefs ?? []).includes(index)
              const agreement = editing.refAgreement?.[index]
              return (
                <div
                  key={url}
                  title={
                    flagged
                      ? `This reference may not be the same person (${agreement}% agreement with the others)`
                      : undefined
                  }
                  className={`group relative aspect-square overflow-hidden rounded border ${
                    flagged ? 'border-amber-500' : 'border-border'
                  }`}
                >
                  <img src={url} alt="" className="h-full w-full object-cover" />
                  <span className="absolute left-0.5 top-0.5 rounded bg-black/70 px-1 text-[9px] text-fg">
                    {index + 1}
                  </span>
                  {editing.refs > 1 && (
                    <button
                      type="button"
                      title="Remove this reference"
                      onClick={() => void removeRef(file, index)}
                      className="absolute right-0.5 top-0.5 hidden h-4 w-4 items-center justify-center rounded bg-black/70 text-muted group-hover:flex hover:text-red-400"
                    >
                      <CloseIcon className="h-2.5 w-2.5" />
                    </button>
                  )}
                </div>
              )
            })}
          </div>
          {(editing.flaggedRefs ?? []).length > 0 && (
            <p className="rounded bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300">
              {(editing.flaggedRefs ?? []).map((i) => i + 1).join(', ')} may not be the same person.
              Scoring matches whichever reference fits best, so a wrong one lets the wrong face
              score well. Remove it, or keep it if it is right.
            </p>
          )}
          <p className="text-[10px] text-muted">
            Drop images here to add more. Order is what the prompt refers to.
          </p>
        </div>

        {(editing.hints ?? []).length > 0 && (
          <div className="space-y-1">
            <span className="text-[10px] uppercase tracking-wide text-muted">
              Stronger character
            </span>
            {/* Each hint resolves to the same action, so make it the button rather than advice
                the user has to go and act on somewhere else. */}
            {(editing.hints ?? []).map((hint) => (
              <button
                key={hint}
                type="button"
                disabled={busy}
                onClick={() =>
                  void pickFilesViaInput().then((f) =>
                    addImages(f.filter((x) => x.type.startsWith('image/'))),
                  )
                }
                className="flex w-full items-center gap-1.5 rounded bg-surface/60 px-2 py-1 text-left text-[10px] text-muted hover:bg-surface hover:text-fg disabled:opacity-40"
              >
                <PlusIcon className="h-3 w-3 shrink-0" />
                {hint}
              </button>
            ))}
          </div>
        )}

        {!editing.faceBearing && (
          <p className="rounded bg-surface/60 px-2 py-1 text-[10px] text-muted">
            No face detected, so continuity is scored on the whole subject rather than the face.
          </p>
        )}

        <div className="border-t border-border pt-2">
          {confirmDelete ? (
            <div className="space-y-1">
              <p className="text-[10px] text-muted">
                Delete {editing.name}? Its references and description go with it. Export it first if
                you want a copy.
              </p>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => void remove(editing.file)}
                  className="rounded bg-red-600 px-2 py-1 text-[11px] text-white hover:bg-red-500"
                >
                  Delete
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="rounded px-2 py-1 text-[11px] text-muted hover:text-fg"
                >
                  Cancel
                </button>
                <a
                  href={`/download/character/${encodeURIComponent(editing.file)}`}
                  download={editing.file}
                  className="ml-auto rounded px-2 py-1 text-[11px] text-muted hover:text-fg"
                >
                  Export
                </a>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmDelete(true)}
              className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] text-muted hover:bg-red-500/10 hover:text-red-300"
            >
              <CloseIcon className="h-3 w-3" />
              Delete character
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
