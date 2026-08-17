import { useEffect, useState } from 'react'
import { useAssetStore } from '../../store/assetStore'
import { useCharacterStore } from '../../store/characterStore'
import { getAssetDragIds, getMediaFileDrag } from '../../lib/dnd'
import { CloseIcon } from '../../components/icons'
import { EncodeProgress } from './EncodeProgress'
import { CharacterBuilds } from './CharacterBuilds'

/** One character's name, description and references; edits are instant and rebuilding is explicit. */
export function CharacterEditor(): React.JSX.Element {
  const { editing, busy, error, rename, setDescription, addRefs, removeRef } = useCharacterStore()
  const [name, setName] = useState(editing?.name ?? '')
  const [description, setDescriptionDraft] = useState(editing?.description ?? '')
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    setName(editing?.name ?? '')
    setDescriptionDraft(editing?.description ?? '')
  }, [editing?.file, editing?.name, editing?.description])

  if (!editing) return <div />

  const file = editing.file
  const nameDirty = name.trim() !== editing.name && name.trim().length > 0
  const descriptionDirty = description !== (editing.description ?? '')

  // Images land in the library first, so a character's references stay ordinary reusable assets.
  const addImages = async (images: File[]): Promise<void> => {
    if (images.length === 0) return
    await useAssetStore.getState().importFiles(images)
    const added = useAssetStore.getState().assets.slice(0, images.length)
    await addRefs(
      file,
      added.map((a) => a.id),
    )
  }

  const addOutput = async (filePath: string): Promise<void> => {
    const response = await fetch(`/media/${filePath}`)
    const blob = await response.blob()
    const name = filePath.split('/').pop() || 'render.png'
    await addImages([new File([blob], name, { type: blob.type || 'image/png' })])
  }

  const onDrop = async (e: React.DragEvent): Promise<void> => {
    e.preventDefault()
    setDragOver(false)
    const assetIds = getAssetDragIds(e.dataTransfer)
    if (assetIds.length > 0) return void addRefs(file, assetIds)
    const media = getMediaFileDrag(e.dataTransfer)
    if (media?.filePath) return void addOutput(media.filePath)
    await addImages(
      Array.from(e.dataTransfer.files ?? []).filter((f) => f.type.startsWith('image/')),
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
      {busy && <EncodeProgress label="Rebuilding…" />}
      {error && <div className="text-[11px] text-red-400">{error}</div>}

      <label className="block space-y-1">
        <span className="text-[10px] uppercase tracking-wide text-muted">Name</span>
        <span className="flex gap-1">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && nameDirty) void rename(file, name.trim())
            }}
            className="min-w-0 flex-1 rounded border border-border bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent"
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
        <span className="text-[10px] uppercase tracking-wide text-muted">Describe character</span>
        <textarea
          value={description}
          onChange={(e) => setDescriptionDraft(e.target.value)}
          rows={3}
          placeholder="Prepended to every prompt that uses this character."
          className="w-full resize-none rounded border border-border bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent"
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

      <div className="space-y-1.5">
        <span className="text-[10px] uppercase tracking-wide text-muted">
          References ({editing.refs})
        </span>
        <div className="grid grid-cols-6 gap-1.5">
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
            Scoring matches whichever reference fits best, so a wrong one lets the wrong face score
            well. Remove it, or keep it if it is right.
          </p>
        )}
        <p className="rounded border border-dashed border-border px-2 py-1.5 text-[10px] text-muted">
          Drag and drop images from your computer, Assets or Outputs.
        </p>
      </div>

      <CharacterBuilds
        file={file}
        builds={editing.builds ?? []}
        description={editing.description ?? ''}
        needsRebuild={editing.needsRebuild ?? false}
      />

      {!editing.faceBearing && (
        <p className="rounded bg-surface/60 px-2 py-1 text-[10px] text-muted">
          No face detected, so continuity is scored on the whole subject rather than the face.
        </p>
      )}
    </div>
  )
}
