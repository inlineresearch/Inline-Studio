/** The dataset editor: a grid of images with per-image captions, plus add + auto-caption controls. */
import { useEffect, useMemo, useState } from 'react'
import type { Asset, TrainingDatasetItem } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { uploadFiles } from '@/lib/importFiles'
import { Modal } from '../../components/Modal'
import { VideoPreview } from '../../components/VideoPreview'
import { useAssetStore } from '../../store/assetStore'
import { useTrainingStore } from '../../store/trainingStore'
import { ipcErrorMessage } from '../../lib/ipcError'

function CaptionBox({
  item,
  datasetId,
}: {
  item: TrainingDatasetItem
  datasetId: string
}): React.JSX.Element {
  const [text, setText] = useState(item.caption)
  const setCaption = useTrainingStore((s) => s.setCaption)
  // `useState(item.caption)` only seeds the FIRST render, so an externally-changed caption (the
  // auto-captioner writing one) would never appear. Re-sync whenever the stored caption changes.
  useEffect(() => setText(item.caption), [item.caption])
  return (
    <textarea
      value={text}
      placeholder="caption…"
      rows={2}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text !== item.caption && void setCaption(datasetId, item.id, text)}
      className="w-full resize-none rounded-b-md border-t border-border bg-black/40 px-2 py-1 text-[11px] text-zinc-200 outline-none focus:bg-black/60"
    />
  )
}

/** A filename without its extension, lowercased, so `1.PNG` pairs with `1.txt`. */
function baseName(name: string): string {
  const stem = name.replace(/\.[^.]+$/, '')
  return stem.toLowerCase()
}

/** Read every dropped/picked `.txt` into a `basename -> caption` map, so it can pair with images. */
async function readCaptionFiles(files: File[]): Promise<Map<string, string>> {
  const captions = new Map<string, string>()
  await Promise.all(
    files
      .filter((f) => f.name.toLowerCase().endsWith('.txt'))
      .map(async (f) => {
        const text = (await f.text()).trim()
        if (text) captions.set(baseName(f.name), text)
      }),
  )
  return captions
}

// Clips are accepted for the archs that can train on them (MiniMax H3). An arch that cannot is
// handed images only by the precache, so a stray clip is skipped rather than breaking a run.
const isMedia = (f: File): boolean =>
  f.type.startsWith('image/') ||
  f.type.startsWith('video/') ||
  /\.(png|jpe?g|webp|bmp|mp4|mov|webm|mkv|avi)$/i.test(f.name)

/** A dataset tile. Video needs a real <video>: the Library defers poster generation, so a clip has
 *  no thumbPath and an <img> pointed at an mp4 renders nothing. */
function Thumb({ asset, src }: { asset?: Asset; src: string }): React.JSX.Element {
  if (asset?.kind === 'video') {
    return <VideoPreview src={src} className="h-full w-full object-cover" />
  }
  return <img src={src} alt="" className="h-full w-full object-cover" />
}

/**
 * The Captioning hub: add or delete images, auto-caption, and edit every caption side by side with
 * its image so long captions are not clipped the way the grid's two-row box clips them. Caption
 * edits are staged and written on Save; add, delete, and auto-caption apply immediately.
 */
function CaptionEditor({
  datasetId,
  items,
  byId,
  busy,
  onAddFiles,
  onClose,
}: {
  datasetId: string
  items: TrainingDatasetItem[]
  byId: Map<string, Asset>
  busy: boolean
  onAddFiles: (files: File[]) => Promise<void>
  onClose: () => void
}): React.JSX.Element {
  const setCaption = useTrainingStore((s) => s.setCaption)
  const autoCaption = useTrainingStore((s) => s.autoCaption)
  const captioning = useTrainingStore((s) => s.captioning)
  const removeItem = useTrainingStore((s) => s.removeItem)
  const removeAll = useTrainingStore((s) => s.removeAll)
  const captioners = useTrainingStore((s) => s.captioners)
  const loadCaptioners = useTrainingStore((s) => s.loadCaptioners)
  // Only edited items get a draft; the rest read live from the store, so a caption added by
  // auto-caption or a freshly imported .txt shows up without reseeding.
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [confirmClear, setConfirmClear] = useState(false)
  const [model, setModel] = useState('')

  useEffect(() => {
    void loadCaptioners()
  }, [loadCaptioners])
  // Reflect the real default (first captioner) in the select once the list arrives.
  useEffect(() => {
    if (!model && captioners.length) setModel(captioners[0].id)
  }, [captioners, model])

  const dirty = items.some((it) => it.id in drafts && drafts[it.id] !== it.caption)

  const save = async (): Promise<void> => {
    setSaving(true)
    try {
      await Promise.all(
        items
          .filter((it) => it.id in drafts && drafts[it.id] !== it.caption)
          .map((it) => setCaption(datasetId, it.id, drafts[it.id])),
      )
      onClose()
    } finally {
      setSaving(false)
    }
  }

  const onPick = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const files = Array.from(e.target.files ?? [])
    e.target.value = ''
    if (files.length) void onAddFiles(files)
  }

  return (
    <Modal open onClose={onClose} title="Captioning">
      <div className="flex flex-col">
        <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-border bg-surface px-4 py-2.5">
          <label
            className={`cursor-pointer rounded-md border border-border px-2.5 py-1 text-xs text-zinc-200 hover:bg-panel ${
              busy ? 'pointer-events-none opacity-40' : ''
            }`}
          >
            {busy ? 'Adding…' : 'Add files'}
            <input
              type="file"
              multiple
              accept="image/*,video/*,.txt"
              className="hidden"
              onChange={onPick}
            />
          </label>
          <button
            onClick={() => void autoCaption(datasetId, false, model || undefined)}
            disabled={captioning || items.length === 0}
            className="rounded-md border border-border px-2.5 py-1 text-xs text-zinc-200 hover:bg-panel disabled:opacity-40"
          >
            {captioning ? 'Captioning…' : 'Auto-caption'}
          </button>
          {captioners.length > 1 && (
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={captioning}
              title="Caption model"
              className="rounded-md border border-border bg-panel px-2 py-1 text-xs text-zinc-200 disabled:opacity-40"
            >
              {captioners.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          )}
          <span className="ml-1 text-[11px] text-zinc-500">{items.length} items</span>
          {items.length > 0 &&
            (confirmClear ? (
              <button
                onClick={() => {
                  void removeAll(datasetId)
                  setConfirmClear(false)
                }}
                className="ml-auto rounded-md border border-red-500/60 px-2.5 py-1 text-xs font-medium text-red-300 hover:bg-red-500/10"
              >
                Confirm delete all
              </button>
            ) : (
              <button
                onClick={() => setConfirmClear(true)}
                className="ml-auto rounded-md border border-border px-2.5 py-1 text-xs text-zinc-400 hover:bg-panel hover:text-zinc-200"
              >
                Delete all
              </button>
            ))}
        </div>

        {items.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-1 px-6 py-16 text-center text-sm text-zinc-500">
            <p>Nothing here yet. Add files to get started.</p>
            <p className="text-xs">
              A .txt next to an image (1.png + 1.txt) is read as its caption.
            </p>
          </div>
        ) : (
          <div className="flex flex-col divide-y divide-border">
            {items.map((item) => {
              const asset = byId.get(item.assetId)
              const src = asset ? resolveMedia(asset.thumbPath ?? asset.filePath) : ''
              return (
                <div key={item.id} className="group flex gap-3 p-4">
                  <div className="relative h-28 w-28 shrink-0 overflow-hidden rounded-md bg-black">
                    {src && <Thumb asset={asset} src={src} />}
                    <button
                      onClick={() => void removeItem(datasetId, item.id)}
                      title="Delete this item"
                      className="absolute right-1 top-1 hidden rounded bg-black/70 px-1.5 py-0.5 text-[11px] text-white group-hover:block"
                    >
                      Delete
                    </button>
                  </div>
                  <textarea
                    value={drafts[item.id] ?? item.caption}
                    placeholder="caption…"
                    rows={4}
                    onChange={(e) => setDrafts((d) => ({ ...d, [item.id]: e.target.value }))}
                    className="min-h-28 w-full resize-y rounded-md border border-border bg-black/40 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-zinc-500"
                  />
                </div>
              )
            })}
          </div>
        )}

        <div className="sticky bottom-0 flex items-center justify-end gap-2 border-t border-border bg-surface px-4 py-3">
          <button
            onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-zinc-300 hover:bg-panel"
          >
            Close
          </button>
          <button
            onClick={() => void save()}
            disabled={saving || !dirty}
            className="rounded-md bg-emerald-500/90 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            {saving ? 'Saving…' : 'Save captions'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

export function DatasetItemsGrid({ datasetId }: { datasetId: string }): React.JSX.Element {
  // `?? []` outside the selector: returning a fresh [] from the selector loops the store (Object.is).
  const items = useTrainingStore((s) => s.itemsByDataset[datasetId]) ?? []
  const addItems = useTrainingStore((s) => s.addItems)
  const setCaption = useTrainingStore((s) => s.setCaption)
  const removeItem = useTrainingStore((s) => s.removeItem)
  const assets = useAssetStore((s) => s.assets)
  const loadAssets = useAssetStore((s) => s.load)
  const setError = useTrainingStore((s) => s.setError)
  const [busy, setBusy] = useState(false)
  // Highlight while OS files are dragged over the grid.
  const [fileOver, setFileOver] = useState(false)

  const [editing, setEditing] = useState(false)

  const byId = useMemo(() => new Map(assets.map((a) => [a.id, a])), [assets])

  /**
   * Import dropped/picked files into the Library and attach them to this dataset. A `.txt` with the
   * same basename as an image (`1.png` + `1.txt`) is read as that image's caption, matching the
   * ComfyUI/kohya dataset convention. Auto-caption then skips anything that already has a caption.
   */
  const attachFiles = async (files: File[]): Promise<void> => {
    setBusy(true)
    try {
      const media = files.filter(isMedia)
      const captions = await readCaptionFiles(files)
      // Captions dropped on their own: pair them to items already in the dataset by filename.
      if (!media.length) {
        await Promise.all(
          items.map((it) => {
            const asset = byId.get(it.assetId)
            const caption = asset ? captions.get(baseName(asset.name)) : undefined
            return caption ? setCaption(datasetId, it.id, caption) : undefined
          }),
        )
        return
      }
      const uploaded = await uploadFiles(media, null)
      if (!uploaded.length) return
      const created = await addItems(
        datasetId,
        uploaded.map((a) => a.id),
      )
      // Pair each new item to its sidecar caption by the image's original filename.
      const itemByAsset = new Map(created.map((it) => [it.assetId, it]))
      await Promise.all(
        uploaded.map((asset) => {
          const caption = captions.get(baseName(asset.name))
          const item = itemByAsset.get(asset.id)
          return caption && item ? setCaption(datasetId, item.id, caption) : undefined
        }),
      )
      await loadAssets() // so thumbnails resolve for the freshly uploaded assets
    } catch (e) {
      setError(ipcErrorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  // True when a drag carries OS files (Finder/Explorer), not an internal asset drag.
  const isFileDrag = (e: React.DragEvent): boolean =>
    Array.from(e.dataTransfer.types).includes('Files')

  const onDragOver = (e: React.DragEvent): void => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    if (!fileOver) setFileOver(true)
  }

  const onDrop = (e: React.DragEvent): void => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    setFileOver(false)
    // Keep images and their .txt captions; readCaptionFiles/attachFiles sort them out.
    const files = Array.from(e.dataTransfer.files ?? []).filter(
      (f) => isMedia(f) || f.name.toLowerCase().endsWith('.txt'),
    )
    if (files.length > 0) void attachFiles(files)
  }

  /** The browser file picker: multi-select images and, optionally, their .txt caption sidecars. */
  const onPick = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const files = Array.from(e.target.files ?? [])
    e.target.value = '' // let the same selection fire change again next time
    if (files.length > 0) void attachFiles(files)
  }

  return (
    <div
      className={`relative flex min-h-0 flex-1 flex-col gap-3 rounded-lg ${
        fileOver ? 'ring-2 ring-inset ring-accent' : ''
      }`}
      onDragOver={onDragOver}
      onDragLeave={(e) => {
        // Only clear when the cursor actually leaves the grid, not a child element.
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFileOver(false)
      }}
      onDrop={onDrop}
    >
      {fileOver && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-lg bg-accent/5">
          <span className="rounded-md border border-accent bg-panel/90 px-3 py-1.5 text-xs text-accent">
            Drop images or clips (and .txt captions) to add to this dataset
          </span>
        </div>
      )}
      <div className="flex items-center gap-2">
        <label
          className={`cursor-pointer rounded-md border border-border px-3 py-1.5 text-sm text-zinc-200 hover:bg-panel ${
            busy ? 'pointer-events-none opacity-40' : ''
          }`}
        >
          {busy ? 'Adding…' : 'Add files'}
          <input
            type="file"
            multiple
            accept="image/*,video/*,.txt"
            className="hidden"
            onChange={onPick}
          />
        </label>
        <button
          onClick={() => setEditing(true)}
          className="rounded-md border border-border px-3 py-1.5 text-sm text-zinc-200 hover:bg-panel"
        >
          Captioning
        </button>
        <span className="text-[11px] text-zinc-500">{items.length} items</span>
      </div>

      {items.length === 0 ? (
        <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-border px-6 text-center text-sm text-zinc-500">
          Drop images here, or use Add files. A .txt next to an image (1.png + 1.txt) is read as its
          caption. 10 to 30 of your character trains a good LoRA.
        </div>
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-2 gap-3 overflow-y-auto pr-1 sm:grid-cols-3 lg:grid-cols-4">
          {items.map((item) => {
            const asset = byId.get(item.assetId)
            const src = asset ? resolveMedia(asset.thumbPath ?? asset.filePath) : ''
            return (
              <div
                key={item.id}
                className="group flex flex-col rounded-md border border-border bg-surface/60"
              >
                <div className="relative aspect-square overflow-hidden rounded-t-md bg-black">
                  {src && <Thumb asset={asset} src={src} />}
                  <button
                    onClick={() => void removeItem(datasetId, item.id)}
                    className="absolute right-1 top-1 hidden rounded bg-black/70 px-1.5 py-0.5 text-[11px] text-white group-hover:block"
                  >
                    Remove
                  </button>
                </div>
                <CaptionBox item={item} datasetId={datasetId} />
              </div>
            )
          })}
        </div>
      )}

      {editing && (
        <CaptionEditor
          datasetId={datasetId}
          items={items}
          byId={byId}
          busy={busy}
          onAddFiles={attachFiles}
          onClose={() => setEditing(false)}
        />
      )}
    </div>
  )
}
