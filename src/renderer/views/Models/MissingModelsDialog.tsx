import { Modal } from '../../components/Modal'
import type { MissingModel, RegistryModel } from '@shared/types'
import { useModelRegistryStore } from '../../store/modelRegistryStore'
import { formatSize } from './formatSize'

/**
 * What a graph asked for that is not here. Every missing file is listed with the folder it belongs
 * in, because that is what lets someone place their own; a download appears only where the registry
 * actually carries the file.
 */
export function MissingModelsDialog(): React.JSX.Element | null {
  const missing = useModelRegistryStore((s) => s.missing)
  const reason = useModelRegistryStore((s) => s.reason)
  const dismiss = useModelRegistryStore((s) => s.dismiss)
  if (!missing) return null

  const downloadable = missing.filter((m) => m.matches.length > 0).length

  return (
    <Modal
      open
      onClose={dismiss}
      title="Models this needs"
      panelClassName="max-h-[80vh] w-full max-w-2xl"
    >
      <div className="space-y-3 p-5">
        <p className="text-[11px] text-muted">
          {reason} {missing.length} file{missing.length === 1 ? '' : 's'}{' '}
          {missing.length === 1 ? 'is' : 'are'} not in your models folder
          {downloadable > 0 ? `, ${downloadable} of which can be downloaded.` : '.'}
        </p>

        {missing.map((row) => (
          <MissingRow key={row.wanted} row={row} />
        ))}
      </div>
    </Modal>
  )
}

function MissingRow({ row }: { row: MissingModel }): React.JSX.Element {
  return (
    <div className="rounded border border-border bg-surface/60 p-2.5">
      <div className="flex items-baseline gap-2">
        <span className="min-w-0 flex-1 truncate text-[11px] text-fg">{row.wanted}</span>
        <span className="shrink-0 font-mono text-[10px] text-muted">{row.path}</span>
      </div>
      {row.matches.length === 0 ? (
        <p className="mt-1 text-[10px] text-muted">
          Not in the model registry. Put the file in the folder above.
        </p>
      ) : (
        <div className="mt-2 space-y-1">
          {row.matches.map((match) => (
            <SourceRow
              key={match.model.id}
              model={match.model}
              exact={match.exact}
              present={match.present}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function SourceRow({
  model,
  exact,
  present,
}: {
  model: RegistryModel
  exact: boolean
  present: boolean
}): React.JSX.Element {
  const download = useModelRegistryStore((s) => s.download)
  const progress = useModelRegistryStore((s) => s.downloading[model.id])

  return (
    <div className="flex items-center gap-2 rounded bg-black/20 px-2 py-1.5">
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[10px] text-fg">
          {exact
            ? model.filename
            : `${model.filename}${model.precision ? ` (${model.precision})` : ''}`}
        </span>
        <span className="block truncate text-[10px] text-muted">
          {model.repo || model.url} {model.verified && '· verified'}
          {model.sizeBytes ? ` · ${formatSize(model.sizeBytes)}` : ''}
        </span>
      </span>
      {present ? (
        <span className="shrink-0 text-[10px] text-emerald-400">Installed</span>
      ) : progress ? (
        <span className="shrink-0 text-[10px] text-muted">
          {Math.round(progress.fraction * 100)}%
        </span>
      ) : (
        <button
          type="button"
          onClick={() => void download(model.id)}
          className="shrink-0 rounded border border-border px-2 py-0.5 text-[10px] text-muted hover:bg-surface hover:text-fg"
        >
          Download
        </button>
      )}
    </div>
  )
}
