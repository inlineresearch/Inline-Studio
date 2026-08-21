import { Modal } from '../../components/Modal'
import type { MissingModel, RegistryModel } from '@shared/types'
import { preferredMatch, useModelRegistryStore } from '../../store/modelRegistryStore'
import { sourceUrl } from './sourceUrl'

/** Folder name to the word people use for it; an unmapped folder shows its own name. */
const TYPE_LABELS: Record<string, string> = {
  diffusion_models: 'Diffusion',
  vae: 'VAE',
  controlnet: 'ControlNet',
  text_encoders: 'Text Encoder',
  loras: 'LoRA',
  clip_vision: 'CLIP Vision',
  upscale_models: 'Upscale',
  embeddings: 'Embedding',
  checkpoints: 'Checkpoint',
}

function typeLabel(path: string): string {
  const folder = path.split('/').filter(Boolean)[0] ?? path
  return TYPE_LABELS[folder] ?? folder
}

/**
 * What a graph asked for that is not here: one row per missing file, with the download offered only
 * where the registry actually carries it.
 */
export function MissingModelsDialog(): React.JSX.Element | null {
  const missing = useModelRegistryStore((s) => s.missing)
  const dismiss = useModelRegistryStore((s) => s.dismiss)
  if (!missing) return null

  return (
    <Modal
      open
      onClose={dismiss}
      title="Missing Models"
      panelClassName="max-h-[70vh] w-full max-w-2xl"
    >
      <div className="flex min-h-0 flex-col gap-3 p-5">
        <p className="text-[11px] text-muted">
          Models required to generate with this node. Downloads land in your models folder; to use
          weights you already have elsewhere, start Inline Studio with{' '}
          <code className="rounded bg-black/40 px-1 font-mono text-[10px] text-zinc-300">
            --models-dir /path/to/models
          </code>
          .
        </p>

        <div className="min-h-0 flex-1 overflow-y-auto">
          <table className="w-full table-fixed text-left">
            <thead className="sticky top-0 bg-surface">
              <tr className="text-[10px] uppercase tracking-wide text-muted">
                <th className="py-1 pr-2 font-normal">File</th>
                <th className="w-32 py-1 pr-2 font-normal">Type</th>
                <th className="w-28 py-1 font-normal">
                  <DownloadAll missing={missing} />
                </th>
              </tr>
            </thead>
            <tbody>
              {missing.map((row) => (
                <MissingRow key={row.wanted} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Modal>
  )
}

/** Queues every row the registry can supply; the rest have nowhere to be fetched from. */
function DownloadAll({ missing }: { missing: MissingModel[] }): React.JSX.Element {
  const downloadAll = useModelRegistryStore((s) => s.downloadAll)
  const downloading = useModelRegistryStore((s) => s.downloading)
  const cancel = useModelRegistryStore((s) => s.cancel)

  const fetchable = missing.filter((m) => preferredMatch(m) && !preferredMatch(m)?.present)
  const active = fetchable.filter((m) => downloading[preferredMatch(m)!.model.id])
  if (fetchable.length === 0) return <span className="font-normal">Download</span>
  if (active.length > 0) {
    return (
      <button
        type="button"
        onClick={() => active.forEach((m) => void cancel(preferredMatch(m)!.model.id))}
        className="rounded px-1.5 py-0.5 text-[10px] font-normal text-muted hover:bg-black/30 hover:text-zinc-200"
      >
        Cancel all
      </button>
    )
  }
  return (
    <button
      type="button"
      onClick={() => void downloadAll()}
      title={`Download all ${fetchable.length}, one at a time`}
      className="rounded bg-emerald-600 px-2 py-0.5 text-[10px] font-normal text-white hover:bg-emerald-500"
    >
      Download all
    </button>
  )
}

function MissingRow({ row }: { row: MissingModel }): React.JSX.Element {
  const match = preferredMatch(row)
  // Only a matched file has a source to open; an unknown one is the user's own to place.
  const link = match ? sourceUrl(match.model) : ''

  return (
    <tr className="border-t border-border/60 text-[11px] text-zinc-300">
      <td className="py-2 pr-2">
        {link ? (
          <a
            href={link}
            target="_blank"
            rel="noreferrer"
            title={`${row.wanted} - open its source`}
            className="block truncate text-fg underline decoration-dotted underline-offset-2 hover:text-accent"
          >
            {row.wanted}
          </a>
        ) : (
          <span className="block truncate text-fg" title={row.wanted}>
            {row.wanted}
          </span>
        )}
      </td>
      <td className="py-2 pr-2 text-[10px] text-muted">{typeLabel(row.path)}</td>
      <td className="py-2">
        {match ? (
          <DownloadCell model={match.model} present={match.present} />
        ) : (
          <span className="text-[10px] text-muted">Not available</span>
        )}
      </td>
    </tr>
  )
}

function DownloadCell({
  model,
  present,
}: {
  model: RegistryModel
  present: boolean
}): React.JSX.Element {
  const download = useModelRegistryStore((s) => s.download)
  const cancel = useModelRegistryStore((s) => s.cancel)
  const progress = useModelRegistryStore((s) => s.downloading[model.id])

  if (present) return <span className="text-[10px] text-emerald-400">Installed</span>
  if (progress) {
    // A queued row has no progress to show, so it says where it is in line instead.
    const waiting = progress.status.startsWith('Queued')
    return (
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-muted">
          {waiting ? progress.status : `${Math.round(progress.fraction * 100)}%`}
        </span>
        <button
          type="button"
          onClick={() => void cancel(model.id)}
          title={waiting ? 'Remove from the queue' : 'Stop this download'}
          className="rounded px-1.5 py-0.5 text-[10px] text-muted hover:bg-black/30 hover:text-zinc-200"
        >
          Cancel
        </button>
      </div>
    )
  }
  return (
    <button
      type="button"
      onClick={() => void download(model.id)}
      className="rounded bg-emerald-600 px-2.5 py-1 text-[10px] text-white hover:bg-emerald-500 disabled:opacity-40"
    >
      Download
    </button>
  )
}
