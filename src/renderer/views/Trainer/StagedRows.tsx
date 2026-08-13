/** The rows a source produced, editable before they are committed to a dataset. */
import { useMemo } from 'react'
import type { Asset, StagedDatasetItem, TrainingMode } from '@shared/types'
import { uploadFiles } from '@/lib/importFiles'
import { LazyMedia } from './LazyMedia'

export function StagedRows({
  rows,
  assets,
  mode,
  onChange,
}: {
  rows: StagedDatasetItem[]
  assets: Asset[]
  mode: TrainingMode
  onChange: (rows: StagedDatasetItem[]) => void
}): React.JSX.Element {
  const byId = useMemo(() => new Map(assets.map((a) => [a.id, a])), [assets])
  const paired = mode === 'control'
  // The trailing column is the delete control's seat, reserved so the prompt never reflows.
  const columns = paired
    ? 'grid-cols-[6.5rem_6.5rem_minmax(0,1fr)_1.5rem]'
    : 'grid-cols-[6.5rem_minmax(0,1fr)_1.5rem]'

  function patch(index: number, next: Partial<StagedDatasetItem>): void {
    onChange(rows.map((r, i) => (i === index ? { ...r, ...next } : r)))
  }

  async function pickReference(index: number, file: File): Promise<void> {
    const [asset] = await uploadFiles([file], null)
    if (asset) patch(index, { referenceAssetId: asset.id })
  }

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed border-border text-center text-xs text-zinc-500">
        Nothing loaded yet. Choose a source above, then Load.
      </div>
    )
  }

  return (
    <div>
      <div className={`grid ${columns} gap-x-2 pb-1 text-[10px] text-zinc-600`}>
        <span>Asset</span>
        {paired && <span>Reference</span>}
        <span>Prompt</span>
        <span />
      </div>

      <div className="flex flex-col gap-2">
        {rows.map((row, index) => (
          <div key={row.assetId} className={`group grid ${columns} items-start gap-2`}>
            <Cell
              asset={byId.get(row.assetId)}
              compare={row.referenceAssetId ? byId.get(row.referenceAssetId) : undefined}
            />
            {paired && (
              <Cell
                asset={row.referenceAssetId ? byId.get(row.referenceAssetId) : undefined}
                compare={byId.get(row.assetId)}
                onPick={(f) => void pickReference(index, f)}
              />
            )}
            <textarea
              value={row.caption}
              placeholder="prompt…"
              rows={3}
              onChange={(e) => patch(index, { caption: e.target.value })}
              className="min-h-[3.6rem] w-full resize-y rounded-md border border-border bg-black/40 px-2 py-1 text-[11px] text-zinc-100 outline-none focus:border-zinc-500"
            />
            <button
              onClick={() => onChange(rows.filter((_, i) => i !== index))}
              title="Drop this row"
              className="mt-1 text-zinc-600 opacity-0 transition hover:text-red-400 group-hover:opacity-100"
            >
              <TrashIcon />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function Cell({
  asset,
  compare,
  onPick,
}: {
  asset?: Asset
  compare?: Asset
  onPick?: (file: File) => void
}): React.JSX.Element {
  if (asset) {
    return (
      <div className="relative aspect-video w-full overflow-hidden rounded border border-border bg-black">
        <LazyMedia asset={asset} compare={compare} />
      </div>
    )
  }
  if (!onPick) {
    return (
      <div className="flex aspect-video w-full items-center justify-center rounded border border-border bg-black text-[10px] text-zinc-600">
        not found
      </div>
    )
  }
  return (
    <label className="flex aspect-video w-full cursor-pointer items-center justify-center rounded border border-dashed border-border bg-black/30 text-[10px] text-zinc-400 hover:border-zinc-500 hover:text-zinc-200">
      Pick
      <input
        type="file"
        accept="image/*,video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          e.target.value = ''
          if (file) onPick(file)
        }}
      />
    </label>
  )
}

/** Lucide-style bin, matching the one the extensions list already uses. */
function TrashIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3.5 w-3.5"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  )
}
