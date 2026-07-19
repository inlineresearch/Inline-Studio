/** The Extensions dialog: what is installed, what is published, and installing from a URL. */
import { useEffect } from 'react'
import { studio } from '@/lib/studio'
import { Modal } from '../../components/Modal'
import { useExtensionsStore, type ExtensionsTab } from '../../store/extensionsStore'
import { ExtensionCard } from './ExtensionCard'
import { InstallPanel } from './InstallPanel'

const TABS: { id: ExtensionsTab; label: string }[] = [
  { id: 'installed', label: 'Installed' },
  { id: 'available', label: 'Available' },
  { id: 'url', label: 'Install from URL' },
]

function Empty({ message }: { message: string }): React.JSX.Element {
  return <div className="px-5 py-10 text-center text-[12px] text-zinc-500">{message}</div>
}

function Installed(): React.JSX.Element {
  const extensions = useExtensionsStore((s) => s.extensions)
  const loading = useExtensionsStore((s) => s.loading)
  if (loading && extensions.length === 0) return <Empty message="Loading…" />
  if (extensions.length === 0) {
    return <Empty message="No extensions installed yet. Browse Available, or install from a URL." />
  }
  return (
    <div className="flex flex-col gap-2.5 p-5">
      {extensions.map((extension) => (
        <ExtensionCard key={extension.extensionId} extension={extension} />
      ))}
    </div>
  )
}

function Available(): React.JSX.Element {
  const registry = useExtensionsStore((s) => s.registry)
  const stale = useExtensionsStore((s) => s.registryStale)
  const installed = useExtensionsStore((s) => s.extensions)
  const beginInstall = useExtensionsStore((s) => s.beginInstall)
  const loadRegistry = useExtensionsStore((s) => s.loadRegistry)

  if (registry.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 px-5 py-10">
        <span className="text-[12px] text-zinc-500">
          {stale ? 'Could not reach the extension registry.' : 'No published extensions yet.'}
        </span>
        <button
          onClick={() => void loadRegistry(true)}
          className="rounded-md border border-border px-3 py-1.5 text-[11px] text-zinc-300 hover:bg-surface"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2.5 p-5">
      {stale && (
        <div className="rounded-md border border-border bg-surface/60 px-3 py-2 text-[11px] text-zinc-400">
          Showing a cached list. The registry could not be reached.
        </div>
      )}
      {registry.map((entry) => {
        const already = installed.some((p) => p.extensionId === entry.id)
        return (
          <div
            key={entry.id}
            className="flex items-start gap-3 rounded-lg border border-border bg-panel/40 p-3"
          >
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-semibold text-zinc-100">{entry.name}</div>
              <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-zinc-500">
                {entry.description}
              </p>
              {entry.author && (
                <div className="mt-1 text-[10px] text-zinc-600">by {entry.author}</div>
              )}
            </div>
            <button
              onClick={() => void beginInstall(entry.repo, entry.pin ?? 'latest')}
              disabled={already}
              className="shrink-0 rounded-md bg-emerald-500/90 px-3 py-1.5 text-[11px] font-semibold text-black hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {already ? 'Installed' : 'Install'}
            </button>
          </div>
        )
      })}
    </div>
  )
}

export function ExtensionsDialog(): React.JSX.Element | null {
  const open = useExtensionsStore((s) => s.open)
  const close = useExtensionsStore((s) => s.closeDialog)
  const tab = useExtensionsStore((s) => s.tab)
  const setTab = useExtensionsStore((s) => s.setTab)
  const restartRequired = useExtensionsStore((s) => s.restartRequired)
  const install = useExtensionsStore((s) => s.install)

  // Subscribed here, not in the canvas: the dialog is reachable with no project open, and progress
  // must still stream.
  useEffect(() => {
    if (!open) return
    return studio().events.onExtensionInstallProgress((e) => {
      useExtensionsStore.getState().onProgress(e)
    })
  }, [open])

  // An install started from the Available tab shows its progress there, not on a hidden tab.
  useEffect(() => {
    if (install?.report || install?.done) setTab('url')
  }, [install?.report, install?.done, setTab])

  if (!open) return null

  return (
    <Modal open={open} onClose={close} title="Extensions">
      {restartRequired && (
        <div className="border-b border-amber-500/30 bg-amber-500/10 px-5 py-2.5 text-[11px] leading-relaxed text-amber-200">
          Restart Inline Studio to finish applying your changes. Python cannot reload code that is
          already running, so an updated or rolled-back extension only takes effect on restart.
        </div>
      )}

      <div className="flex gap-1 border-b border-border px-4 pt-3">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-t-md px-3 py-1.5 text-[12px] font-medium transition-colors ${
              tab === t.id
                ? 'border-b-2 border-emerald-400 text-zinc-100'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'installed' && <Installed />}
      {tab === 'available' && <Available />}
      {tab === 'url' && <InstallPanel />}
    </Modal>
  )
}
