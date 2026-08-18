import { Modal } from '../../components/Modal'
import type { MissingModel, RegistryModel } from '@shared/types'
import { useModelRegistryStore } from '../../store/modelRegistryStore'
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

/** The registry may carry the same file at several precisions; an exact filename match wins. */
function preferredMatch(row: MissingModel): MissingModel['matches'][number] | null {
  return row.matches.find((m) => m.exact) ?? row.matches[0] ?? null
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
                <th className="w-28 py-1 font-normal">Download</th>
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
  const progress = useModelRegistryStore((s) => s.downloading[model.id])

  if (present) return <span className="text-[10px] text-emerald-400">Installed</span>
  if (progress)
    return <span className="text-[10px] text-muted">{Math.round(progress.fraction * 100)}%</span>
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
