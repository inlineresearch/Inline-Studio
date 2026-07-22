/**
 * The Trainer tab: pick/create a dataset, caption its images, set hyperparameters, and run a LoRA
 * training job with live progress + samples + host telemetry. The produced LoRA lands in
 * models/loras/, so it shows up in the Studio tab's LoRA loader node.
 */
import { useEffect, useState } from 'react'
import { subscribeTrainingEvents, useTrainingStore } from '../../store/trainingStore'
import { DatasetIcon } from '../../components/icons'
import { DatasetItemsGrid } from './DatasetItemsGrid'
import { HyperparamForm } from './HyperparamForm'
import { TrainingMonitor } from './TrainingMonitor'
import { SystemStatsBar } from './SystemStatsBar'

function NewDataset(): React.JSX.Element {
  const [name, setName] = useState('')
  const [trigger, setTrigger] = useState('')
  const create = useTrainingStore((s) => s.createDataset)
  const onCreate = async (): Promise<void> => {
    if (!name.trim()) return
    await create(name.trim(), trigger.trim())
    setName('')
    setTrigger('')
  }
  return (
    <div className="flex flex-col gap-1.5 border-t border-border p-2">
      <input
        value={name}
        placeholder="New dataset name"
        onChange={(e) => setName(e.target.value)}
        className="rounded border border-border bg-black/30 px-2 py-1 text-xs text-zinc-100 outline-none focus:border-zinc-500"
      />
      <input
        value={trigger}
        placeholder="Trigger word (e.g. ohwx)"
        onChange={(e) => setTrigger(e.target.value)}
        className="rounded border border-border bg-black/30 px-2 py-1 text-xs text-zinc-100 outline-none focus:border-zinc-500"
      />
      <button
        onClick={() => void onCreate()}
        className="rounded bg-emerald-600 py-1 text-xs font-medium text-white hover:bg-emerald-500"
      >
        Create dataset
      </button>
    </div>
  )
}

export function TrainerPanel(): React.JSX.Element {
  const datasets = useTrainingStore((s) => s.datasets)
  const activeId = useTrainingStore((s) => s.activeDatasetId)
  const runs = useTrainingStore((s) => s.runs)
  const error = useTrainingStore((s) => s.error)
  const load = useTrainingStore((s) => s.loadDatasets)
  const loadRuns = useTrainingStore((s) => s.loadRuns)
  const select = useTrainingStore((s) => s.selectDataset)

  useEffect(() => {
    void load()
    void loadRuns()
    return subscribeTrainingEvents()
  }, [load, loadRuns])

  const active = datasets.find((d) => d.id === activeId) ?? null
  const activeRunning = runs.some((r) => r.status === 'training' || r.status === 'queued')
  const datasetRuns = runs.filter((r) => r.datasetId === activeId)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex min-h-0 flex-1">
        {/* Dataset sidebar */}
        <div className="flex w-56 shrink-0 flex-col border-r border-border bg-surface/40">
          <div className="flex items-center gap-2 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
            <DatasetIcon className="h-4 w-4" /> Datasets
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {datasets.map((d) => (
              <button
                key={d.id}
                onClick={() => select(d.id)}
                className={`flex w-full flex-col items-start px-3 py-2 text-left text-sm ${
                  d.id === activeId ? 'bg-panel text-white' : 'text-zinc-300 hover:bg-panel/60'
                }`}
              >
                <span className="truncate">{d.name}</span>
                {d.triggerWord && (
                  <span className="text-[10px] text-zinc-500">{d.triggerWord}</span>
                )}
              </button>
            ))}
          </div>
          <NewDataset />
        </div>

        {/* Dataset editor */}
        <div className="flex min-w-0 flex-1 flex-col gap-3 p-4">
          {error && (
            <div className="rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</div>
          )}
          {active ? (
            <DatasetItemsGrid datasetId={active.id} />
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-zinc-500">
              Create or select a dataset to begin.
            </div>
          )}
        </div>

        {/* Settings + runs */}
        {active && (
          <div className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto border-l border-border p-4">
            <HyperparamForm datasetId={active.id} disabled={activeRunning} />
            {datasetRuns.map((run) => (
              <TrainingMonitor key={run.id} run={run} />
            ))}
          </div>
        )}
      </div>
      <SystemStatsBar />
    </div>
  )
}
