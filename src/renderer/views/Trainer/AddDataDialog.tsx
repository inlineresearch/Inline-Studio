/** Adding training data: where it comes from, what kind of LoRA it is, and what is in it so far. */
import { useEffect, useState } from 'react'
import type { Asset, DatasetRepoPreview, StagedDatasetItem, TrainingMode } from '@shared/types'
import { Modal } from '../../components/Modal'
import { useTrainingStore } from '../../store/trainingStore'
import { StagedRows } from './StagedRows'

type Source = 'machine' | 'path' | 'huggingface'

const SOURCES: { id: Source; label: string }[] = [
  { id: 'machine', label: 'This machine' },
  { id: 'path', label: 'Path' },
  { id: 'huggingface', label: 'Hugging Face' },
]

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`
  return `${Math.round(bytes / 1e3)} KB`
}

export function AddDataDialog({
  datasetId,
  assets,
  onClose,
}: {
  datasetId: string
  assets: Asset[]
  onClose: () => void
}): React.JSX.Element {
  const setDatasetMode = useTrainingStore((s) => s.setDatasetMode)
  const stageFromPath = useTrainingStore((s) => s.stageFromPath)
  const stageFromRepo = useTrainingStore((s) => s.stageFromRepo)
  const commitStaged = useTrainingStore((s) => s.commitStaged)
  const stageFiles = useTrainingStore((s) => s.stageFiles)
  const items = useTrainingStore((s) => s.itemsByDataset[datasetId])
  const captionAssets = useTrainingStore((s) => s.captionAssets)
  const captioners = useTrainingStore((s) => s.captioners)
  const loadCaptioners = useTrainingStore((s) => s.loadCaptioners)
  const captioning = useTrainingStore((s) => s.captioning)
  const inspectDatasetRepo = useTrainingStore((s) => s.inspectDatasetRepo)
  const inspectDatasetPath = useTrainingStore((s) => s.inspectDatasetPath)
  const mode = useTrainingStore((s) => s.datasets.find((d) => d.id === datasetId)?.mode ?? 'clip')

  const [source, setSource] = useState<Source>('machine')
  const [busy, setBusy] = useState(false)
  const [path, setPath] = useState('')
  const [repo, setRepo] = useState('')
  const [preview, setPreview] = useState<DatasetRepoPreview | null>(null)
  const [checking, setChecking] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  // Rows a source produced. They exist as assets but as no dataset row until Import.
  const [staged, setStaged] = useState<StagedDatasetItem[]>([])
  const [captioner, setCaptioner] = useState('')
  // What the dataset held when the dialog opened, to tell an edit from a no-op.
  const [original, setOriginal] = useState('')

  async function check(): Promise<void> {
    setChecking(true)
    setPreview(null)
    try {
      setPreview(
        source === 'path'
          ? await inspectDatasetPath(path.trim())
          : await inspectDatasetRepo(repo.trim()),
      )
    } finally {
      setChecking(false)
    }
  }

  /** Pull the chosen source into the review list below. Nothing reaches the dataset yet. */
  async function load(): Promise<void> {
    setBusy(true)
    try {
      const rows =
        source === 'machine'
          ? await stageFiles(files)
          : source === 'path'
            ? await stageFromPath(path.trim())
            : await stageFromRepo(repo.trim())
      if (rows) setStaged((prev) => [...prev, ...rows])
      setFiles([])
      setPreview(null)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void loadCaptioners()
  }, [loadCaptioners])
  // The same button manages an existing dataset, so the dialog opens holding what is already in
  // it. Import then reconciles, which is what lets a row be dropped here and leave the dataset.
  useEffect(() => {
    const rows = (items ?? []).map((it) => ({
      assetId: it.assetId,
      name: '',
      caption: it.caption,
      referenceAssetId: it.referenceAssetId,
    }))
    setStaged(rows)
    setOriginal(fingerprint(rows))
    // Only on open: re-syncing on every store change would discard edits mid-session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId])
  useEffect(() => {
    if (!captioner && captioners.length) setCaptioner(captioners[0].id)
  }, [captioner, captioners])

  // A dataset arrives part-captioned more often than not: `dataset.json` and `.txt` sidecars both
  // populate prompts at staging. Captioning those again would spend minutes of VLM time to
  // overwrite something better than it produces.
  const uncaptioned = staged.filter((r) => !r.caption.trim())
  const dirty = fingerprint(staged) !== original

  /** Caption the staged rows that have no prompt, before any of them reach the dataset. */
  async function caption(): Promise<void> {
    const written = await captionAssets(
      uncaptioned.map((r) => r.assetId),
      captioner || undefined,
    )
    setStaged((rows) =>
      rows.map((r) => (written[r.assetId] ? { ...r, caption: written[r.assetId] } : r)),
    )
  }

  /** Accept the staged rows into the dataset. The first and only write. */
  async function accept(): Promise<void> {
    setBusy(true)
    try {
      await commitStaged(datasetId, staged)
      setOriginal(fingerprint(staged))
      onClose()
    } finally {
      setBusy(false)
    }
  }

  const loadLabel = (): string => {
    if (busy) return source === 'huggingface' ? 'Pulling…' : 'Loading…'
    if (source === 'machine') return `Load ${files.length} files`
    return preview && !preview.problem ? `Load ${preview.items} items` : 'Load'
  }

  return (
    <Modal
      open
      onClose={onClose}
      title="Add training data"
      panelClassName="h-[calc(100vh-200px)] w-full max-w-5xl"
      bodyClassName="flex min-h-0 flex-1 flex-col"
      headerAction={
        <button
          onClick={() => void accept()}
          disabled={busy || !dirty}
          title={
            dirty ? `Make the dataset match these ${staged.length} rows` : 'Nothing has changed'
          }
          className="rounded-md border border-accent px-3 py-1 text-xs font-medium text-accent transition hover:bg-accent/10 disabled:cursor-not-allowed disabled:border-border disabled:text-zinc-600"
        >
          {busy ? 'Importing…' : `Import ${staged.length}`}
        </button>
      }
    >
      <div className="flex min-h-0 flex-1 flex-col">
        {/* ---- Source, and what kind of LoRA this dataset is ---- */}
        <div className="border-b border-border px-4 pb-3 pt-3">
          <div className="flex items-center justify-between gap-4">
            <p className="text-xs font-medium text-zinc-300">Load dataset from</p>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-zinc-500">LoRA type</span>
              <div className="flex overflow-hidden rounded-md border border-border">
                {(['clip', 'control'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => void setDatasetMode(datasetId, m as TrainingMode)}
                    className={`px-3 py-1 text-xs transition ${
                      mode === m
                        ? 'bg-accent/15 text-accent'
                        : 'text-zinc-400 hover:bg-panel hover:text-zinc-200'
                    }`}
                  >
                    {m === 'clip' ? 'Clip' : 'Control'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-2 flex gap-1">
            {SOURCES.map((s) => (
              <button
                key={s.id}
                onClick={() => {
                  setSource(s.id)
                  setPreview(null)
                }}
                className={`rounded-md px-3 py-1 text-xs transition ${
                  source === s.id
                    ? 'bg-accent/15 text-accent'
                    : 'text-zinc-500 hover:bg-panel/60 hover:text-zinc-300'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          {/* The row below the tabs is whatever the chosen source needs. */}
          <div className="mt-2 min-h-[2.25rem]">
            {source === 'machine' && (
              <label
                className={`inline-flex cursor-pointer items-center rounded-md border border-border px-3 py-1.5 text-xs text-zinc-200 hover:bg-panel ${
                  busy ? 'pointer-events-none opacity-40' : ''
                }`}
              >
                {files.length ? `${files.length} files chosen` : 'Choose files'}
                <input
                  type="file"
                  multiple
                  accept="image/*,video/*,.txt"
                  className="hidden"
                  onChange={(e) => {
                    const picked = [...(e.target.files ?? [])]
                    e.target.value = ''
                    if (picked.length) setFiles(picked)
                  }}
                />
              </label>
            )}
            {source === 'machine' && files.length > 0 && (
              <button
                onClick={() => void load()}
                disabled={busy}
                className="ml-2 rounded-md border border-accent px-3 py-1.5 text-xs font-medium text-accent hover:bg-accent/10 disabled:opacity-40"
              >
                {loadLabel()}
              </button>
            )}

            {source === 'path' && (
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  <input
                    value={path}
                    onChange={(e) => {
                      setPath(e.target.value)
                      setPreview(null)
                    }}
                    placeholder="/path/to/a/folder of clips"
                    spellCheck={false}
                    className="flex-1 rounded-md border border-border bg-black/30 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-zinc-500"
                  />
                  <button
                    onClick={() => void check()}
                    disabled={checking || busy || !path.trim()}
                    className="rounded-md border border-border px-3 py-1.5 text-xs text-zinc-200 hover:bg-panel disabled:opacity-40"
                  >
                    {checking ? 'Checking…' : 'Check'}
                  </button>
                </div>
                {preview && (
                  <PreviewRow preview={preview} busy={busy} label={loadLabel()} onLoad={load} />
                )}
              </div>
            )}

            {source === 'huggingface' && (
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  <input
                    value={repo}
                    onChange={(e) => {
                      setRepo(e.target.value)
                      setPreview(null)
                    }}
                    placeholder="Lightricks/Canny-Control-Dataset"
                    spellCheck={false}
                    className="flex-1 rounded-md border border-border bg-black/30 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-zinc-500"
                  />
                  <button
                    onClick={() => void check()}
                    disabled={checking || busy || !repo.trim()}
                    className="rounded-md border border-border px-3 py-1.5 text-xs text-zinc-200 hover:bg-panel disabled:opacity-40"
                  >
                    {checking ? 'Checking…' : 'Check'}
                  </button>
                </div>
                {preview && (
                  <PreviewRow preview={preview} busy={busy} label={loadLabel()} onLoad={load} />
                )}
              </div>
            )}
          </div>

          <p className="mt-2 text-[10px] text-zinc-600">
            Pairs are detected automatically: dataset.json or metadata.jsonl if present, otherwise
            bear.mp4 is matched with bear_reference.mp4.
          </p>
        </div>

        {/* ---- What is in the dataset. The only thing that scrolls. ---- */}
        <div className="flex min-h-0 flex-1 flex-col px-4 pb-2 pt-3">
          <div className="flex items-center gap-2 pb-2">
            <p className="text-xs font-medium text-zinc-300">To import</p>
            <span className="ml-auto text-[11px] text-zinc-500">
              {mode === 'control'
                ? `${staged.filter((r) => r.referenceAssetId).length}/${staged.length} paired`
                : `${staged.length} items`}
            </span>
            <span className="text-[11px] text-zinc-500">Auto caption</span>
            <select
              value={captioner}
              onChange={(e) => setCaptioner(e.target.value)}
              className="rounded-md border border-border bg-black/30 px-1.5 py-0.5 text-[11px] text-zinc-200 outline-none focus:border-zinc-500"
            >
              {captioners.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
            <button
              onClick={() => void caption()}
              disabled={uncaptioned.length === 0 || captioning || busy}
              title={
                staged.length === 0
                  ? 'Load something first'
                  : uncaptioned.length === 0
                    ? 'Every row already has a prompt'
                    : `Caption the ${uncaptioned.length} rows with no prompt`
              }
              className="rounded-md border border-border px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-panel disabled:cursor-not-allowed disabled:text-zinc-600"
            >
              {captioning
                ? 'Captioning…'
                : uncaptioned.length
                  ? `Caption ${uncaptioned.length}`
                  : 'Caption'}
            </button>
            {staged.length > 0 && (
              <button
                onClick={() => setStaged([])}
                title={`Drop all ${staged.length} staged rows`}
                className="rounded-md border border-border px-2 py-0.5 text-[11px] text-zinc-400 hover:bg-panel hover:text-zinc-200"
              >
                Delete {staged.length}
              </button>
            )}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            <StagedRows rows={staged} assets={assets} mode={mode} onChange={setStaged} />
          </div>
        </div>
      </div>
    </Modal>
  )
}

/** What the chosen source holds, and the button that pulls it into the dataset section. */
function PreviewRow({
  preview,
  busy,
  label,
  onLoad,
}: {
  preview: DatasetRepoPreview
  busy: boolean
  label: string
  onLoad: () => Promise<void>
}): React.JSX.Element {
  if (preview.problem) {
    return <div className="px-0.5 text-xs text-red-400">{preview.problem}</div>
  }
  return (
    <div className="flex items-center gap-3 px-0.5 text-xs">
      <span className="text-zinc-400">
        {preview.items} items
        {preview.pairs > 0 ? `, ${preview.pairs} paired` : ', no references'} ·{' '}
        {formatBytes(preview.bytes)}
        {preview.metadataFile && ` · ${preview.metadataFile}`}
      </span>
      <button
        onClick={() => void onLoad()}
        disabled={busy}
        className="ml-auto rounded-md border border-accent px-3 py-1 text-xs font-medium text-accent hover:bg-accent/10 disabled:opacity-40"
      >
        {label}
      </button>
    </div>
  )
}

/** The staged rows as one comparable string: order, asset, reference and prompt all count. */
function fingerprint(rows: StagedDatasetItem[]): string {
  return rows.map((r) => `${r.assetId}|${r.referenceAssetId ?? ''}|${r.caption}`).join('\n')
}
