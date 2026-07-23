/**
 * The Trainer node's Adjust sidebar - hyperparameters live here, never on the node face (the same
 * split the Generate / Inline Core nodes use). Edits persist into the node's `data.hyperparams`.
 */
import type { TrainingBaseMode, TrainingHyperparams } from '@shared/types'
import { useTrainerBoardStore } from '../../store/trainerBoardStore'
import { useTrainingStore } from '../../store/trainingStore'
import { XIcon } from '../Moodboard/nodes/NodeBadge'

const DEFAULTS: TrainingHyperparams = {
  baseMode: 'deturbo',
  rank: 16,
  alpha: 16,
  learningRate: 1e-4,
  steps: 1500,
  batchSize: 1,
  resolution: 1024,
  saveEvery: 250,
  gpuIds: [],
}

function NumberField({
  label,
  value,
  step,
  onChange,
}: {
  label: string
  value: number
  step?: number
  onChange: (v: number) => void
}): React.JSX.Element {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
      {label}
      <input
        type="number"
        value={value}
        step={step ?? 1}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
      />
    </label>
  )
}

export function TrainerSettingsPanel({ itemId }: { itemId: string }): React.JSX.Element | null {
  const item = useTrainerBoardStore((s) => s.items.find((i) => i.id === itemId))
  const patchData = useTrainerBoardStore((s) => s.patchData)
  const toggleSettings = useTrainerBoardStore((s) => s.toggleSettings)
  const gpus = useTrainingStore((s) => s.systemStats?.gpus) ?? []
  if (!item) return null

  const hp: TrainingHyperparams = { ...DEFAULTS, ...(item.data.hyperparams ?? {}) }
  const set = <K extends keyof TrainingHyperparams>(key: K, value: TrainingHyperparams[K]): void =>
    void patchData(itemId, { hyperparams: { ...hp, [key]: value } })

  const toggleGpu = (index: number): void =>
    set(
      'gpuIds',
      hp.gpuIds.includes(index) ? hp.gpuIds.filter((g) => g !== index) : [...hp.gpuIds, index],
    )

  return (
    <div className="flex w-80 shrink-0 flex-col gap-3 overflow-y-auto border-l border-border bg-surface/40 p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-200">Training settings</span>
        <button
          onClick={() => toggleSettings(itemId)}
          className="rounded p-1 text-zinc-400 hover:bg-panel hover:text-zinc-200"
          title="Close"
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </div>

      <label className="flex flex-col gap-1 text-[11px] text-zinc-400">
        Base
        <select
          value={hp.baseMode}
          onChange={(e) => set('baseMode', e.target.value as TrainingBaseMode)}
          className="rounded-md border border-border bg-black/30 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-zinc-500"
        >
          <option value="turbo_adapter">Z-Image Turbo (+ training adapter)</option>
          <option value="deturbo">Z-Image De-Turbo</option>
        </select>
      </label>

      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Rank" value={hp.rank} onChange={(v) => set('rank', v)} />
        <NumberField label="Alpha" value={hp.alpha} onChange={(v) => set('alpha', v)} />
        <NumberField
          label="Learning rate"
          value={hp.learningRate}
          step={0.00001}
          onChange={(v) => set('learningRate', v)}
        />
        <NumberField label="Steps" value={hp.steps} step={100} onChange={(v) => set('steps', v)} />
        <NumberField
          label="Batch size"
          value={hp.batchSize}
          onChange={(v) => set('batchSize', v)}
        />
        <NumberField
          label="Resolution"
          value={hp.resolution}
          step={64}
          onChange={(v) => set('resolution', v)}
        />
        <NumberField
          label="Save every"
          value={hp.saveEvery}
          step={50}
          onChange={(v) => set('saveEvery', v)}
        />
      </div>

      {gpus.length > 1 && (
        <div className="flex flex-col gap-1 text-[11px] text-zinc-400">
          GPUs (leave empty for auto)
          <div className="flex flex-wrap gap-2">
            {gpus.map((g) => (
              <button
                key={g.index}
                onClick={() => toggleGpu(g.index)}
                className={`rounded-md border px-2 py-1 text-[11px] ${
                  hp.gpuIds.includes(g.index)
                    ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300'
                    : 'border-border text-zinc-400 hover:bg-panel'
                }`}
              >
                GPU {g.index}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
