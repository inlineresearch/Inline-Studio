/**
 * "Load Dataset" node: picks a training dataset and feeds it downstream. The node face stays a
 * preview (thumbnails + counts); the images/captions themselves are edited in the side panel that
 * opens when the node is selected - keeping heavy editing off the card, like every other node.
 */
import { useEffect, useRef, useState } from 'react'
import { type NodeProps } from '@xyflow/react'
import { PortHandle } from '../../Moodboard/nodes/PortHandle'
import { topStyle } from '../../Moodboard/nodes/nodeSize'
import { PairTile } from '../PairTile'
import { useAssetStore } from '../../../store/assetStore'
import { useTrainingStore } from '../../../store/trainingStore'
import { NodeFrame } from '../../Moodboard/nodes/NodeFrame'
import { LayersIcon, NodeBadge, NodeBadgeRow } from '../../Moodboard/nodes/NodeBadge'
import { DATASET_HANDLE } from './handles'
import { useBoardActions } from '../../Moodboard/nodes/boardActions'

export function TrainDatasetNode({ id, selected }: NodeProps): React.JSX.Element {
  const { items: board, patchData } = useBoardActions()
  const item = board.find((i) => i.id === id)
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
  const captioned = items.filter((it) => it.caption.trim()).length
  const clips = items.filter((it) => byId.get(it.assetId)?.kind === 'video').length
  const noun = clips === 0 ? 'images' : clips === items.length ? 'clips' : 'items'

  // The face is a preview, so it draws only what fits and grows as the node is resized. A fixed
  // count both wasted a large card and, on a clip dataset, would open a video decoder per hidden
  // tile. The footer always reports the true total, so this never reads as the whole dataset.
  const gridRef = useRef<HTMLDivElement>(null)
  const [capacity, setCapacity] = useState(6)
  useEffect(() => {
    const el = gridRef.current
    if (!el) return
    const CELL = 40 // 36px tile + 4px gap
    const measure = (): void =>
      setCapacity(
        Math.max(1, Math.floor(el.clientWidth / CELL)) *
          Math.max(1, Math.floor(el.clientHeight / CELL)),
      )
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  const thumbs = items.slice(0, capacity)

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
          <div ref={gridRef} className="flex-1 overflow-hidden bg-black p-1">
            {items.length === 0 ? (
              <div className="flex h-full items-center justify-center text-[11px] text-zinc-600">
                {datasetId ? 'No items yet' : 'Pick a dataset'}
              </div>
            ) : (
              <div
                className="grid gap-1"
                style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(36px, 1fr))' }}
              >
                {thumbs.map((it) => (
                  <PairTile
                    key={it.id}
                    target={byId.get(it.assetId)}
                    reference={it.referenceAssetId ? byId.get(it.referenceAssetId) : undefined}
                  />
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-border bg-surface/90 px-2 py-1">
            <span className="text-[10px] text-zinc-500">
              {items.length} {noun} · {captioned} captioned
            </span>
          </div>
        </div>
      </NodeFrame>
      <PortHandle
        id={DATASET_HANDLE}
        label="Dataset"
        kind="dataset"
        side="output"
        style={topStyle(0)}
      />
    </>
  )
}
