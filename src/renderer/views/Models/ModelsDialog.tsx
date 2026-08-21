import { useEffect, useMemo, useState } from 'react'
import { Modal } from '../../components/Modal'
import type { RegistryModel } from '@shared/types'
import { useModelRegistryStore } from '../../store/modelRegistryStore'
import { formatSize } from './formatSize'

/** Every published model, what is already here, and where each one lands. */
export function ModelsDialog({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}): React.JSX.Element | null {
  const entries = useModelRegistryStore((s) => s.entries)
  const stale = useModelRegistryStore((s) => s.stale)
  const loading = useModelRegistryStore((s) => s.loading)
  const error = useModelRegistryStore((s) => s.error)
  const load = useModelRegistryStore((s) => s.load)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    if (open) void load()
  }, [open, load])

  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase()
    const rows = needle
      ? entries.filter((e) =>
          [e.filename, e.label, e.repo, e.category].some((f) => f.toLowerCase().includes(needle)),
        )
      : entries
    return [...rows].sort((a, b) =>
      a.category === b.category
        ? a.filename.localeCompare(b.filename)
        : a.category.localeCompare(b.category),
    )
  }, [entries, filter])

  if (!open) return null
  const here = entries.filter((e) => e.present).length

  return (
    <Modal
      open
      onClose={onClose}
      title="Models"
      panelClassName="max-h-[85vh] w-full max-w-4xl"
      headerAction={
        <button
          type="button"
          onClick={() => void load(true)}
          className="rounded border border-border px-2 py-1 text-[10px] text-muted hover:text-fg"
        >
          Refresh
        </button>
      }
    >
      <div className="flex min-h-0 flex-col gap-2 p-5">
        <div className="flex items-center gap-2">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by name, repo or folder"
            className="min-w-0 flex-1 rounded border border-border bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent"
          />
          <span className="shrink-0 text-[10px] text-muted">
            {here} of {entries.length} installed
          </span>
        </div>

        {error && <p className="text-[11px] text-red-400">{error}</p>}
        {stale && (
          <p className="rounded bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300">
            The registry could not be reached, so this list is the one cached on this machine.
          </p>
        )}
        {loading && entries.length === 0 && <p className="text-[11px] text-muted">Loading…</p>}

        <div className="min-h-0 flex-1 overflow-y-auto">
          <table className="w-full text-left">
            <thead className="sticky top-0 bg-surface">
              <tr className="text-[10px] uppercase tracking-wide text-muted">
                <th className="py-1 pr-2 font-normal">File</th>
                <th className="py-1 pr-2 font-normal">Folder</th>
                <th className="py-1 pr-2 font-normal">Type</th>
                <th className="py-1 pr-2 font-normal">Size</th>
                <th className="py-1 pr-2 font-normal">Source</th>
                <th className="py-1 font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((model) => (
                <ModelRow key={model.id} model={model} />
              ))}
            </tbody>
          </table>
          {shown.length === 0 && !loading && (
            <p className="p-4 text-center text-[11px] text-muted">Nothing matches that.</p>
          )}
        </div>
      </div>
    </Modal>
  )
}

function ModelRow({ model }: { model: RegistryModel }): React.JSX.Element {
  const download = useModelRegistryStore((s) => s.download)
  const progress = useModelRegistryStore((s) => s.downloading[model.id])

  return (
    <tr className="border-t border-border/60 text-[11px] text-zinc-300">
      <td className="py-1.5 pr-2">
        <span className="block truncate text-fg">{model.filename}</span>
        {model.precision && <span className="text-[10px] text-muted">{model.precision}</span>}
      </td>
      <td className="py-1.5 pr-2 font-mono text-[10px] text-muted">{model.category}</td>
      <td className="py-1.5 pr-2 text-[10px] text-muted">
        {model.kind === 'hf_folder' ? 'folder' : model.kind === 'url' ? 'url' : 'file'}
      </td>
      <td className="py-1.5 pr-2 text-[10px] text-muted">
        {model.sizeBytes ? formatSize(model.sizeBytes) : '—'}
      </td>
      <td className="py-1.5 pr-2">
        <span className="block truncate text-[10px] text-muted">{model.repo || model.url}</span>
        {model.verified && <span className="text-[10px] text-emerald-400">verified</span>}
      </td>
      <td className="py-1.5">
        {model.present ? (
          <span className="text-[10px] text-emerald-400">Installed</span>
        ) : progress ? (
          <span className="text-[10px] text-muted">{Math.round(progress.fraction * 100)}%</span>
        ) : (
          <button
            type="button"
            onClick={() => void download(model.id)}
            className="rounded border border-border px-2 py-0.5 text-[10px] text-muted hover:bg-surface hover:text-fg"
          >
            Download
          </button>
        )}
      </td>
    </tr>
  )
}
