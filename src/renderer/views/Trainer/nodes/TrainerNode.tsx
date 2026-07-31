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
import { useCoreNodesStore } from '../../../store/coreNodesStore'
import { activeDownload, useModelRequirementsStore } from '../../../store/modelRequirementsStore'
import { NodeFrame } from '../../Moodboard/nodes/NodeFrame'
import {
  AdjustIcon,
  AlertIcon,
  NodeBadge,
  NodeBadgeRow,
  WandIcon,
} from '../../Moodboard/nodes/NodeBadge'
import { NodeRunToolbar } from '../../Moodboard/nodes/NodeRunToolbar'
import { DATASET_HANDLE, RUN_HANDLE, wiredDatasetId } from './handles'

/** The training arch + base maps to the generation node type whose weights it trains on, so the
 * Trainer node can reuse that node's requirements check + download flow. */
function requirementType(arch: string, baseMode: string): string {
  if (arch === 'krea2')
    return baseMode === 'turbo_adapter' ? 'krea/krea-2-turbo' : 'krea/krea-2-raw'
  if (arch === 'flux2') return 'black-forest-labs/flux-2'
  return 'alibaba/z-image-turbo'
}

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

  // Follow the tail, but only while the user is already at the bottom - otherwise scrolling back
  // through the log would be yanked to the end every time a new line streams in.
  const logRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)
  useEffect(() => {
    const el = logRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [logs.length])

  // Another node holding the GPU blocks this one (the backend allows one run at a time).
  const otherRunning = runs.some(
    (r) => r.id !== runId && (r.status === 'training' || r.status === 'queued'),
  )
  const control = controlFor(run)
  const hp = { ...DEFAULT_HP, ...(item?.data.hyperparams ?? {}) }

  // Same "missing models" hint the Generate / Core nodes show: resolve the base this run needs to
  // its generation node type, check its requirements, and blink a chip that opens the download popup.
  const arch = (item?.data.hyperparams as { arch?: string } | undefined)?.arch ?? 'z-image'
  const reqType = requirementType(arch, hp.baseMode)
  const registryVersion = useCoreNodesStore((s) => s.registryVersion)
  const loadReqs = useModelRequirementsStore((s) => s.load)
  const openReqs = useModelRequirementsStore((s) => s.open)
  const reqs = useModelRequirementsStore((s) => s.byType[reqType])
  const downloadsForType = useModelRequirementsStore((s) => s.downloads[reqType])
  useEffect(() => {
    void loadReqs(reqType)
  }, [reqType, registryVersion, loadReqs])
  const modelsMissing = reqs ? !reqs.allPresent : false
  const download = downloadsForType ? activeDownload(downloadsForType, reqs) : null

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
        {(modelsMissing || download) && (
          <button
            onClick={() => openReqs(reqType)}
            title={download ? 'Downloading base model…' : 'Base model missing - click to download'}
            className={`nodrag flex h-6 items-center gap-1 rounded-full border px-2 text-[10px] font-medium shadow-sm backdrop-blur ${
              download
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'animate-pulse border-amber-500/40 bg-amber-500/10 text-amber-300 hover:animate-none hover:bg-amber-500/20'
            }`}
          >
            <AlertIcon className="h-3.5 w-3.5" />
            {download ? `${Math.round(download.fraction * 100)}%` : 'Base model'}
          </button>
        )}
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
            {/* `nowheel` stops React Flow swallowing the wheel to zoom the canvas, so the log
                actually scrolls; `nodrag` keeps a drag-select inside the log from moving the node. */}
            <div
              ref={logRef}
              onScroll={(e) => {
                const el = e.currentTarget
                stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
              }}
              className={`nowheel nodrag h-full overflow-auto px-2 pb-2 font-mono text-[10px] leading-snug text-zinc-300 ${
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
