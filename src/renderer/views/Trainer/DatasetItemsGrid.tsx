/** The dataset editor: a grid of images with per-image captions, plus add + auto-caption controls. */
import { useMemo, useState } from 'react'
import type { TrainingDatasetItem } from '@shared/types'
import { resolveMedia } from '@/lib/media'
import { studio } from '@/lib/studio'
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
  const items = useTrainingStore((s) => s.itemsByDataset[datasetId] ?? [])
  const captioning = useTrainingStore((s) => s.captioning)
  const addItems = useTrainingStore((s) => s.addItems)
  const removeItem = useTrainingStore((s) => s.removeItem)
  const autoCaption = useTrainingStore((s) => s.autoCaption)
  const assets = useAssetStore((s) => s.assets)
  const loadAssets = useAssetStore((s) => s.load)
  const setError = useTrainingStore((s) => s.setError)
  const [busy, setBusy] = useState(false)

  const byId = useMemo(() => new Map(assets.map((a) => [a.id, a])), [assets])

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
    <div className="flex min-h-0 flex-1 flex-col gap-3">
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
          Add 10–30 images of your character to train a LoRA.
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
