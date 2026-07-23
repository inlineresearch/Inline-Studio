/**
 * "Train LoRA" node: runs the training job for a wired dataset and reports live progress + logs.
 *
 * The floating run chip is a state machine over the run's status - Start / Stop / Resume Training -
 * so a cancelled run (which flushes a checkpoint) is picked back up rather than restarted. The bound
 * `runId` is persisted in the node's data, so the node rebinds to its run after a reload and can
 * still offer Resume. Hyperparameters live in the Adjust sidebar, never on the node face.
 */
import { useEffect, useRef } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { TrainingRun } from '@shared/types'
import { useTrainingStore } from '../../../store/trainingStore'
import { useTrainerBoardStore } from '../../../store/trainerBoardStore'
import { NodeFrame } from '../../Moodboard/nodes/NodeFrame'
import { AdjustIcon, NodeBadge, NodeBadgeRow, WandIcon } from '../../Moodboard/nodes/NodeBadge'
import { NodeRunToolbar } from '../../Moodboard/nodes/NodeRunToolbar'
import { DATASET_HANDLE, RUN_HANDLE, wiredDatasetId } from './handles'

const DEFAULT_HP = {
  baseMode: 'deturbo' as const,
  rank: 16,
  alpha: 16,
  learningRate: 1e-4,
  steps: 1500,
  batchSize: 1,
  resolution: 1024,
  saveEvery: 250,
  gpuIds: [] as number[],
}

/** Run / Stop / Resume, derived from the bound run's status. */
type Control = 'run' | 'stop' | 'resume'

function controlFor(run: TrainingRun | null): Control {
  if (!run) return 'run'
  if (run.status === 'training' || run.status === 'queued') return 'stop'
  if (run.status === 'interrupted' || run.status === 'failed') return 'resume'
  return 'run'
}

export function TrainerNode({ id, selected }: NodeProps): React.JSX.Element {
  const item = useTrainerBoardStore((s) => s.items.find((i) => i.id === id))
  const items = useTrainerBoardStore((s) => s.items)
  const connectors = useTrainerBoardStore((s) => s.connectors)
  const patchData = useTrainerBoardStore((s) => s.patchData)
  const toggleSettings = useTrainerBoardStore((s) => s.toggleSettings)
  const settingsItemId = useTrainerBoardStore((s) => s.settingsItemId)

  const runs = useTrainingStore((s) => s.runs)
  const loadRuns = useTrainingStore((s) => s.loadRuns)
  const start = useTrainingStore((s) => s.start)
  const cancel = useTrainingStore((s) => s.cancel)
  const resume = useTrainingStore((s) => s.resume)
  const datasets = useTrainingStore((s) => s.datasets)

  const wired = wiredDatasetId(id, connectors, items)
  const own = (item?.data.datasetId as string | null | undefined) ?? null
  const datasetId = wired ?? own
  const dataset = datasets.find((d) => d.id === datasetId) ?? null
  const runId = (item?.data.runId as string | null | undefined) ?? null
  const run = runs.find((r) => r.id === runId) ?? null
  const progress = useTrainingStore((s) => (runId ? s.progressByRun[runId] : undefined))
  const logs = useTrainingStore((s) => (runId ? s.logsByRun[runId] : undefined)) ?? []

  useEffect(() => {
    void loadRuns()
  }, [loadRuns])

  // Follow the tail: a training run streams continuously, so keep the newest line in view.
  const logRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logs.length])

  // Another node holding the GPU blocks this one (the backend allows one run at a time).
  const otherRunning = runs.some(
    (r) => r.id !== runId && (r.status === 'training' || r.status === 'queued'),
  )
  const control = controlFor(run)
  const hp = { ...DEFAULT_HP, ...(item?.data.hyperparams ?? {}) }
  const fraction = progress?.fraction ?? run?.progressFraction ?? 0
  const step = progress?.step ?? run?.step ?? 0
  const totalSteps = progress?.totalSteps || run?.totalSteps || hp.steps

  const onControl = async (): Promise<void> => {
    if (control === 'stop' && runId) return void cancel(runId)
    if (control === 'resume' && runId) return void resume(runId)
    if (!datasetId) return
    const created = await start(datasetId, hp)
    // Persist the run so this node rebinds to it after a reload (and can offer Resume).
    if (created) void patchData(id, { runId: created.id })
  }

  const busy = control === 'stop'
  const statusLabel = run ? run.status : 'idle'

  return (
    <>
      {/* The same floating Run chip the Generate / Core nodes use - right-aligned above the node.
          Visible while training even when unselected, so a run can always be stopped. */}
      <NodeRunToolbar
        isTarget={!!selected || busy}
        busy={busy}
        onRun={() => void onControl()}
        onStop={() => void onControl()}
        disabled={!datasetId || (otherRunning && !busy)}
        disabledReason={
          !datasetId ? 'Wire a dataset first' : 'Another training run is using the GPU'
        }
        runLabel={control === 'resume' ? 'Resume Training' : 'Start Training'}
        stopLabel="Stop Training"
      />
      <NodeBadgeRow dragNodeId={id}>
        <NodeBadge icon={<WandIcon />}>Train LoRA</NodeBadge>
        <NodeBadge tone="info" accent={busy ? 'text-emerald-400' : undefined}>
          rank {hp.rank}
        </NodeBadge>
      </NodeBadgeRow>
      <NodeFrame id={id} selected={!!selected} padded={false} subtleSelect minWidth={260}>
        <div className="flex h-full flex-col">
          <div className="relative flex-1 overflow-hidden bg-black">
            {busy && (
              <span className="absolute left-2 top-2 z-10 rounded-full bg-black/70 px-2 py-0.5 text-[10px] text-emerald-300">
                {progress?.status ?? 'training'} · {step}/{totalSteps}
              </span>
            )}
            {/* Logs: the trainer's streamed stdout, newest last. Strictly ONE entry per line - no
                wrapping, so a line never reflows into several as the node is resized. Overflow
                scrolls horizontally; widening the node just reveals more of each line. */}
            <div
              ref={logRef}
              className={`h-full overflow-auto px-2 pb-2 font-mono text-[10px] leading-snug text-zinc-300 ${
                busy ? 'pt-8' : 'pt-2'
              }`}
            >
              {logs.length === 0 ? (
                <span className="text-zinc-500">
                  {dataset ? `Ready · ${dataset.name}` : 'Wire a dataset to train'}
                </span>
              ) : (
                logs.slice(-200).map((line, i) => (
                  <div key={i} className="whitespace-pre">
                    {line}
                  </div>
                ))
              )}
            </div>
            {busy && (
              <div className="absolute bottom-0 left-0 h-px w-full bg-zinc-800">
                <div
                  className="h-full bg-emerald-500 transition-[width] duration-500"
                  style={{ width: `${Math.min(100, fraction * 100)}%` }}
                />
              </div>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-border bg-surface/90 px-2 py-1">
            <span className="truncate text-[10px] text-zinc-500">
              {statusLabel}
              {run ? ` · ${step}/${totalSteps}` : ''}
            </span>
            <button
              data-gen-settings-toggle
              onClick={() => toggleSettings(id)}
              title="Adjust training settings"
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
        id={RUN_HANDLE}
        className="group !h-3 !w-3 !border-2 !border-surface !bg-violet-400"
        title="Run"
      />
    </>
  )
}
