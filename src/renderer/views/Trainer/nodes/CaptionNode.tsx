/**
 * Caption node: auto-captions a dataset's images with the local VLM. Takes a wired dataset (a
 * wired handle wins over its own picker, matching how the Z-Image node treats wired components),
 * runs the captioner, and passes the dataset through so a Trainer can chain off it.
 */
import { useEffect } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTrainingStore } from '../../../store/trainingStore'
import { useTrainerBoardStore } from '../../../store/trainerBoardStore'
import { NodeFrame } from '../../Moodboard/nodes/NodeFrame'
import { AdjustIcon, CaptionGlyph, NodeBadge, NodeBadgeRow } from '../../Moodboard/nodes/NodeBadge'
import { NodeRunToolbar } from '../../Moodboard/nodes/NodeRunToolbar'
import { DATASET_HANDLE, wiredDatasetId } from './handles'

export function CaptionNode({ id, selected }: NodeProps): React.JSX.Element {
  const item = useTrainerBoardStore((s) => s.items.find((i) => i.id === id))
  const items = useTrainerBoardStore((s) => s.items)
  const connectors = useTrainerBoardStore((s) => s.connectors)
  const patchData = useTrainerBoardStore((s) => s.patchData)
  const datasets = useTrainingStore((s) => s.datasets)
  const loadDatasets = useTrainingStore((s) => s.loadDatasets)
  const loadItems = useTrainingStore((s) => s.loadItems)
  const autoCaption = useTrainingStore((s) => s.autoCaption)
  const captioning = useTrainingStore((s) => s.captioning)
  const toggleSettings = useTrainerBoardStore((s) => s.toggleSettings)
  const settingsItemId = useTrainerBoardStore((s) => s.settingsItemId)

  useEffect(() => {
    void loadDatasets()
  }, [loadDatasets])

  // A wired dataset overrides the node's own picker (the Z-Image "wired handle wins" pattern).
  const wired = wiredDatasetId(id, connectors, items)
  const own = (item?.data.datasetId as string | null | undefined) ?? null
  const datasetId = wired ?? own
  const overwrite = Boolean(item?.data.overwrite)
  const dsItems =
    useTrainingStore((s) => (datasetId ? s.itemsByDataset[datasetId] : undefined)) ?? []
  const progress = useTrainingStore((s) => (datasetId ? s.captionProgress[datasetId] : undefined))
  const dataset = datasets.find((d) => d.id === datasetId) ?? null

  useEffect(() => {
    if (datasetId) void loadItems(datasetId)
  }, [datasetId, loadItems])

  const captioned = dsItems.filter((it) => it.caption.trim()).length
  const total = dsItems.length
  const pending = overwrite ? total : total - captioned

  return (
    <>
      {/* The same floating control every runnable node uses, rather than one buried in the footer. */}
      <NodeRunToolbar
        isTarget={!!selected}
        busy={captioning}
        onRun={() => datasetId && void autoCaption(datasetId, overwrite)}
        onStop={() => {}}
        disabled={!datasetId || total === 0}
        disabledReason={!datasetId ? 'Wire or pick a dataset first' : 'This dataset has no images'}
        runLabel="Auto-caption"
      />
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<CaptionGlyph />}>Caption</NodeBadge>
        {total > 0 && (
          <NodeBadge tone="info">
            {captioned}/{total}
          </NodeBadge>
        )}
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect minWidth={220}>
        <div className="flex h-full flex-col">
          {!wired && (
            <div className="border-b border-border p-2">
              <select
                value={own ?? ''}
                onChange={(e) => void patchData(id, { datasetId: e.target.value || null })}
                className="nodrag w-full rounded border border-border bg-black/30 px-2 py-1 text-xs text-zinc-100 outline-none focus:border-zinc-500"
              >
                <option value="">Select a dataset…</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="relative flex flex-1 flex-col items-center justify-center gap-1 bg-black px-3">
            <span className="text-[11px] text-zinc-300">
              {dataset ? dataset.name : 'No dataset'}
            </span>
            <span className="text-[10px] text-zinc-500">
              {total === 0
                ? 'Nothing to caption'
                : pending > 0
                  ? `${pending} image${pending === 1 ? '' : 's'} to caption`
                  : 'All images captioned'}
            </span>
            {(captioning || progress) && (
              <>
                <span className="absolute left-2 top-2 rounded-full bg-black/70 px-2 py-0.5 text-[10px] text-emerald-300">
                  {progress ? `Captioning ${progress.done}/${progress.total}` : 'Captioning…'}
                </span>
                {/* Determinate once the first tick lands; an indeterminate pulse while the
                    captioner model is still loading (no ticks yet). */}
                <div className="absolute bottom-0 left-0 h-px w-full bg-zinc-800">
                  <div
                    className={`h-full bg-emerald-500 ${progress ? 'transition-[width] duration-300' : 'animate-pulse'}`}
                    style={{
                      width: progress?.total
                        ? `${Math.min(100, (progress.done / progress.total) * 100)}%`
                        : '100%',
                    }}
                  />
                </div>
              </>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-border bg-surface/90 px-2 py-1">
            {/* The model is configurable (INLINE_CAPTIONER_MODEL), so don't name one here. */}
            <span className="text-[10px] text-zinc-500">Auto-caption</span>
            <button
              data-gen-settings-toggle
              onClick={() => toggleSettings(id)}
              title="Adjust caption settings"
              className={`nodrag rounded p-1 hover:bg-panel ${
                settingsItemId === id ? 'text-zinc-100' : 'text-zinc-400'
              }`}
            >
              <AdjustIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      </NodeFrame>
      <Handle
        type="target"
        position={Position.Left}
        id={DATASET_HANDLE}
        className="group !h-3 !w-3 !border-2 !border-surface !bg-sky-400"
        title="Dataset"
      />
      <Handle
        type="source"
        position={Position.Right}
        id={DATASET_HANDLE}
        className="group !h-3 !w-3 !border-2 !border-surface !bg-sky-400"
        title="Dataset"
      />
    </>
  )
}
