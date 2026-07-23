/** The dataset editor: a grid of images with per-image captions, plus add + auto-caption controls. */
import { useEffect, useMemo, useState } from 'react'
import type { TrainingDatasetItem } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { studio } from '@/lib/studio'
import { uploadFiles } from '@/lib/importFiles'
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

export function DatasetItemsGrid({ datasetId }: { datasetId: string }): React.JSX.Element {
  // `?? []` outside the selector: returning a fresh [] from the selector loops the store (Object.is).
  const items = useTrainingStore((s) => s.itemsByDataset[datasetId]) ?? []
  const captioning = useTrainingStore((s) => s.captioning)
  const addItems = useTrainingStore((s) => s.addItems)
  const removeItem = useTrainingStore((s) => s.removeItem)
  const autoCaption = useTrainingStore((s) => s.autoCaption)
  const assets = useAssetStore((s) => s.assets)
  const loadAssets = useAssetStore((s) => s.load)
  const setError = useTrainingStore((s) => s.setError)
  const [busy, setBusy] = useState(false)
  // Highlight while OS files are dragged over the grid.
  const [fileOver, setFileOver] = useState(false)

  const byId = useMemo(() => new Map(assets.map((a) => [a.id, a])), [assets])

  /** Import dropped/picked files into the Library, then attach them to this dataset. */
  const attachFiles = async (files: File[]): Promise<void> => {
    setBusy(true)
    try {
      const uploaded = await uploadFiles(files, null)
      if (uploaded.length) {
        await addItems(
          datasetId,
          uploaded.map((a) => a.id),
        )
        await loadAssets() // so thumbnails resolve for the freshly uploaded assets
      }
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
    const files = Array.from(e.dataTransfer.files ?? []).filter((f) => f.type.startsWith('image/'))
    if (files.length > 0) void attachFiles(files)
  }

  const onAdd = async (): Promise<void> => {
    setBusy(true)
    try {
      const res = await studio().assets.importDialog(null)
      if (res.ok && res.value.length) {
        await addItems(
          datasetId,
          res.value.map((a) => a.id),
        )
        await loadAssets() // so thumbnails resolve for the freshly uploaded assets
      }
    } catch (e) {
      setError(ipcErrorMessage(e))
    } finally {
      setBusy(false)
    }
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
            Drop images to add to this dataset
          </span>
        </div>
      )}
      <div className="flex items-center gap-2">
        <button
          onClick={() => void onAdd()}
          disabled={busy}
          className="rounded-md border border-border px-3 py-1.5 text-sm text-zinc-200 hover:bg-panel disabled:opacity-40"
        >
          {busy ? 'Adding…' : 'Add images'}
        </button>
        <button
          onClick={() => void autoCaption(datasetId, false)}
          disabled={captioning || items.length === 0}
          className="rounded-md border border-border px-3 py-1.5 text-sm text-zinc-200 hover:bg-panel disabled:opacity-40"
        >
          {captioning ? 'Captioning…' : 'Auto-caption'}
        </button>
        <span className="text-[11px] text-zinc-500">{items.length} images</span>
      </div>

      {items.length === 0 ? (
        <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed border-border text-sm text-zinc-500">
          Drop images here, or use Add images — 10–30 of your character trains a good LoRA.
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
                  {src && <img src={src} alt="" className="h-full w-full object-cover" />}
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
    </div>
  )
}
