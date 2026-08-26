/** Training datasets in the side rail: pick one, edit its pairs, or create another. */
import { useEffect, useState } from 'react'
import type { TrainingDataset } from '@shared/types'
import { EditIcon } from '../../components/icons'
import { useTrainingStore } from '../../store/trainingStore'
import { DatasetItemsGrid } from './DatasetItemsGrid'

const FIELD =
  'rounded border border-border bg-black/30 px-2 py-1 text-xs text-zinc-100 outline-none focus:border-zinc-500'

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
        className={FIELD}
      />
      <input
        value={trigger}
        placeholder="Trigger word (e.g. ohwx)"
        onChange={(e) => setTrigger(e.target.value)}
        className={FIELD}
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

function DatasetRow({ dataset }: { dataset: TrainingDataset }): React.JSX.Element {
  const activeId = useTrainingStore((s) => s.activeDatasetId)
  const select = useTrainingStore((s) => s.selectDataset)
  const update = useTrainingStore((s) => s.updateDataset)
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(dataset.name)
  const [trigger, setTrigger] = useState(dataset.triggerWord)
  const active = dataset.id === activeId

  const onEdit = (): void => {
    setName(dataset.name)
    setTrigger(dataset.triggerWord)
    setEditing(true)
  }
  const onSave = async (): Promise<void> => {
    if (!name.trim()) return
    await update(dataset.id, name.trim(), trigger.trim())
    setEditing(false)
  }
  const onKeyDown = (e: React.KeyboardEvent): void => {
    if (e.key === 'Enter') void onSave()
    if (e.key === 'Escape') setEditing(false)
  }

  if (editing) {
    return (
      <div className="flex flex-col gap-1.5 bg-panel px-2 py-2">
        <input
          autoFocus
          value={name}
          placeholder="Dataset name"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={onKeyDown}
          className={FIELD}
        />
        <input
          value={trigger}
          placeholder="Trigger word (e.g. ohwx)"
          onChange={(e) => setTrigger(e.target.value)}
          onKeyDown={onKeyDown}
          className={FIELD}
        />
        {/* The trigger is prepended at export, so a change only reaches runs started after it. */}
        <p className="text-[10px] leading-tight text-zinc-500">
          The trigger applies to the next run. Finished runs keep the one they trained with.
        </p>
        <div className="flex gap-1.5">
          <button
            onClick={() => void onSave()}
            className="flex-1 rounded bg-emerald-600 py-1 text-xs font-medium text-white hover:bg-emerald-500"
          >
            Save
          </button>
          <button
            onClick={() => setEditing(false)}
            className="flex-1 rounded border border-border py-1 text-xs text-zinc-300 hover:border-zinc-500"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={`group relative ${active ? 'bg-panel' : 'hover:bg-panel/60'}`}>
      <button
        onClick={() => select(dataset.id)}
        className={`flex w-full flex-col items-start px-3 py-2 pr-8 text-left text-sm ${
          active ? 'text-white' : 'text-zinc-300'
        }`}
      >
        <span className="w-full truncate">{dataset.name}</span>
        {dataset.triggerWord && (
          <span className="text-[10px] text-zinc-500">{dataset.triggerWord}</span>
        )}
      </button>
      <button
        type="button"
        title="Rename, or change the trigger word"
        onClick={onEdit}
        className="absolute right-1.5 top-1.5 hidden h-5 w-5 items-center justify-center rounded bg-black/60 text-zinc-400 hover:text-white group-hover:flex"
      >
        <EditIcon className="h-3 w-3" />
      </button>
    </div>
  )
}

export function DatasetsPanel(): React.JSX.Element {
  const datasets = useTrainingStore((s) => s.datasets)
  const activeId = useTrainingStore((s) => s.activeDatasetId)
  const error = useTrainingStore((s) => s.error)
  const load = useTrainingStore((s) => s.loadDatasets)
  const loadRuns = useTrainingStore((s) => s.loadRuns)

  useEffect(() => {
    void load()
    void loadRuns()
  }, [load, loadRuns])

  const active = datasets.find((d) => d.id === activeId) ?? null

  return (
    <div className="flex h-full min-h-0 flex-col">
      {error && (
        <div className="mx-3 mt-2 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">
          {error}
        </div>
      )}
      <div className="max-h-40 shrink-0 overflow-y-auto border-b border-border">
        {datasets.map((d) => (
          <DatasetRow key={d.id} dataset={d} />
        ))}
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
        {active ? (
          <DatasetItemsGrid datasetId={active.id} />
        ) : (
          <div className="flex flex-1 items-center justify-center text-center text-sm text-zinc-500">
            Select a dataset, or add a Load Dataset node.
          </div>
        )}
      </div>
      <NewDataset />
    </div>
  )
}
