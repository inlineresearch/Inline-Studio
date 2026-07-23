/**
 * The Trainer tab: a dataset panel on the left, the training node graph in the centre, and a node's
 * Adjust sidebar on the right. Nodes (Load Dataset → Caption → Train LoRA → Graph, plus the utility
 * Resource node) live on the `trainer` canvas surface. The produced LoRA lands in models/loras/, so
 * it shows up in the Studio tab's LoRA loader node.
 */
import { useEffect, useState } from 'react'
import { useTrainingStore } from '../../store/trainingStore'
import { useTrainerBoardStore } from '../../store/trainerBoardStore'
import { DatasetIcon, LoraOutputIcon } from '../../components/icons'
import { ChevronLeftIcon, ChevronRightIcon } from '../Moodboard/nodes/NodeBadge'
import { DatasetItemsGrid } from './DatasetItemsGrid'
import { TrainerCanvas } from './TrainerCanvas'
import { TrainerSettingsPanel } from './TrainerSettingsPanel'
import { OutputsPanel } from './OutputsPanel'

type SideTab = 'datasets' | 'outputs'

const TABS: {
  key: SideTab
  label: string
  Icon: (p: { className?: string }) => React.JSX.Element
}[] = [
  { key: 'datasets', label: 'Datasets', Icon: DatasetIcon },
  { key: 'outputs', label: 'Outputs', Icon: LoraOutputIcon },
]

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
  const error = useTrainingStore((s) => s.error)
  const load = useTrainingStore((s) => s.loadDatasets)
  const loadRuns = useTrainingStore((s) => s.loadRuns)
  const select = useTrainingStore((s) => s.selectDataset)
  const settingsItemId = useTrainerBoardStore((s) => s.settingsItemId)
  // The dataset panel collapses so the canvas can take the full width.
  const [panelOpen, setPanelOpen] = useState(true)
  const [tab, setTab] = useState<SideTab>('datasets')

  useEffect(() => {
    void load()
    void loadRuns()
  }, [load, loadRuns])

  const active = datasets.find((d) => d.id === activeId) ?? null

  return (
    <div className="flex h-full min-h-0">
      {/* Collapsed: an icon rail, mirroring the Studio SideMenu. */}
      {!panelOpen && (
        <div className="flex w-11 shrink-0 flex-col items-center gap-1 border-r border-border bg-surface py-2">
          <button
            onClick={() => setPanelOpen(true)}
            title="Expand menu"
            aria-label="Expand menu"
            className="mb-1 flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-panel hover:text-white"
          >
            <ChevronRightIcon className="h-5 w-5" />
          </button>
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => {
                setTab(t.key)
                setPanelOpen(true)
              }}
              title={t.label}
              aria-pressed={tab === t.key}
              className={`flex h-9 w-9 items-center justify-center rounded-md transition-colors ${
                tab === t.key
                  ? 'bg-accent text-panel shadow-sm'
                  : 'text-zinc-400 hover:bg-panel hover:text-zinc-200'
              }`}
            >
              <t.Icon className="h-5 w-5" />
            </button>
          ))}
        </div>
      )}

      {panelOpen && (
        <div className="flex w-[22rem] shrink-0 flex-col border-r border-border bg-surface">
          {/* Segmented tab control + collapse, matching the Studio side menu's header row. */}
          <div className="flex items-center gap-1 border-b border-border px-1.5 py-1.5">
            <div className="flex min-w-0 flex-1 items-center gap-0.5 rounded-lg bg-black/20 p-0.5">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  aria-pressed={tab === t.key}
                  className={`flex min-w-0 flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                    tab === t.key
                      ? 'bg-accent text-panel shadow-sm'
                      : 'text-zinc-400 hover:bg-panel hover:text-zinc-200'
                  }`}
                >
                  <t.Icon className="h-4 w-4 shrink-0" />
                  <span className="truncate">{t.label}</span>
                </button>
              ))}
            </div>
            <button
              onClick={() => setPanelOpen(false)}
              title="Collapse menu"
              aria-label="Collapse menu"
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-zinc-400 hover:bg-panel hover:text-white"
            >
              <ChevronLeftIcon className="h-5 w-5" />
            </button>
          </div>

          {error && (
            <div className="mx-3 mt-2 rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">
              {error}
            </div>
          )}

          {tab === 'outputs' ? (
            <OutputsPanel />
          ) : (
            <>
              <div className="max-h-40 shrink-0 overflow-y-auto border-b border-border">
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
            </>
          )}
        </div>
      )}

      {/* The training graph. */}
      <div className="relative min-w-0 flex-1">
        <TrainerCanvas />
      </div>

      {/* A node's Adjust sidebar (params live off the node face). */}
      {settingsItemId && <TrainerSettingsPanel itemId={settingsItemId} />}
    </div>
  )
}
