/**
 * "Load Dataset" node: picks a training dataset and feeds it downstream. The node face stays a
 * preview (thumbnails + counts); the images/captions themselves are edited in the side panel that
 * opens when the node is selected - keeping heavy editing off the card, like every other node.
 */
import { useEffect } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { resolveMedia } from '@/lib/media'
import { useAssetStore } from '../../../store/assetStore'
import { useTrainingStore } from '../../../store/trainingStore'
import { useTrainerBoardStore } from '../../../store/trainerBoardStore'
import { NodeFrame } from '../../Moodboard/nodes/NodeFrame'
import { LayersIcon, NodeBadge, NodeBadgeRow } from '../../Moodboard/nodes/NodeBadge'
import { DATASET_HANDLE } from './handles'

export function TrainDatasetNode({ id, selected }: NodeProps): React.JSX.Element {
  const item = useTrainerBoardStore((s) => s.items.find((i) => i.id === id))
  const patchData = useTrainerBoardStore((s) => s.patchData)
  const datasets = useTrainingStore((s) => s.datasets)
  const loadDatasets = useTrainingStore((s) => s.loadDatasets)
  const loadItems = useTrainingStore((s) => s.loadItems)
  const selectDataset = useTrainingStore((s) => s.selectDataset)
  const assets = useAssetStore((s) => s.assets)

  const datasetId = (item?.data.datasetId as string | null | undefined) ?? null
  const items = useTrainingStore((s) => (datasetId ? s.itemsByDataset[datasetId] : undefined)) ?? []
  const dataset = datasets.find((d) => d.id === datasetId) ?? null

  useEffect(() => {
    void loadDatasets()
  }, [loadDatasets])

  useEffect(() => {
    if (datasetId) void loadItems(datasetId)
  }, [datasetId, loadItems])

  // Selecting the node makes its dataset the one the side panel edits.
  useEffect(() => {
    if (selected && datasetId) selectDataset(datasetId)
  }, [selected, datasetId, selectDataset])

  const byId = new Map(assets.map((a) => [a.id, a]))
  const thumbs = items.slice(0, 6)
  const captioned = items.filter((it) => it.caption.trim()).length

  return (
    <>
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<LayersIcon />}>Load Dataset</NodeBadge>
        {dataset?.triggerWord && (
          <NodeBadge tone="info" accent="text-emerald-400" title="Trigger word">
            {dataset.triggerWord}
          </NodeBadge>
        )}
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect minWidth={220}>
        <div className="flex h-full flex-col">
          <div className="border-b border-border p-2">
            <select
              value={datasetId ?? ''}
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
          <div className="flex-1 overflow-hidden bg-black p-1">
            {thumbs.length === 0 ? (
              <div className="flex h-full items-center justify-center text-[11px] text-zinc-600">
                {datasetId ? 'No images yet' : 'Pick a dataset'}
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-1">
                {thumbs.map((it) => {
                  const asset = byId.get(it.assetId)
                  const src = asset ? resolveMedia(asset.thumbPath ?? asset.filePath) : ''
                  return (
                    <div
                      key={it.id}
                      className="aspect-square overflow-hidden rounded-sm bg-zinc-900"
                    >
                      {src && <img src={src} alt="" className="h-full w-full object-cover" />}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-border bg-surface/90 px-2 py-1">
            <span className="text-[10px] text-zinc-500">
              {items.length} images · {captioned} captioned
            </span>
          </div>
        </div>
      </NodeFrame>
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
