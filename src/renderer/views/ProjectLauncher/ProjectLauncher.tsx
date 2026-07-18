import { useEffect, useState } from 'react'
import { studio } from '@/lib/studio'
import { Logo } from '../../components/Logo'
import { useProjectStore } from '../../store/projectStore'
import { useUpdateStore } from '../../store/updateStore'
import { DemoSampleCard } from './DemoSampleCard'
import { GettingStartedCard } from './GettingStartedCard'
import { FalKeyCard } from './FalKeyCard'

export function ProjectLauncher(): React.JSX.Element {
  const {
    recents,
    loading,
    error,
    notice,
    exportingPath,
    createProject,
    openFromDialog,
    openFromZip,
    openByPath,
    exportProject,
  } = useProjectStore()
  const [name, setName] = useState('')

  const currentVersion = useUpdateStore((s) => s.currentVersion)
  const updateStatus = useUpdateStore((s) => s.status)
  const updateVersion = useUpdateStore((s) => s.version)
  const loadCurrentVersion = useUpdateStore((s) => s.loadCurrentVersion)
  const openReleases = useUpdateStore((s) => s.openReleases)
  const updateAvailable = updateStatus !== 'idle'

  // Show the running version, and re-check for a newer GitHub release while the user
  // is on the launcher (the auto-updater also checks at startup; this keeps it fresh).
  useEffect(() => {
    void loadCurrentVersion()
    void studio().updates.check()
  }, [loadCurrentVersion])

  const canCreate = name.trim().length > 0 && !loading
  const exporting = exportingPath !== null

  return (
    <div className="relative flex h-full flex-col overflow-y-auto p-8">
      <div className="flex flex-1 items-center">
        <div className="mx-auto w-full max-w-4xl">
          <header className="mb-8 text-center">
            <div className="flex items-center justify-center gap-3">
              <Logo size={44} />
              <h1 className="text-4xl font-semibold tracking-tight text-white">Inline Studio</h1>
            </div>
            <p className="mt-3 text-sm text-zinc-400">
              AI filmmaking on a node canvas, powered by your own ComfyUI
            </p>
          </header>

          <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-2">
            {/* Left column - your projects */}
            <div className="flex flex-col gap-6">
              <section className="rounded-xl border border-border bg-surface p-5">
                <h2 className="mb-3 text-sm font-medium text-zinc-300">New project</h2>
                <div className="flex gap-2">
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && canCreate) void createProject(name.trim())
                    }}
                    placeholder="Untitled film"
                    className="flex-1 rounded-lg border border-border bg-panel px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500 focus:border-accent"
                  />
                  <button
                    disabled={!canCreate}
                    onClick={() => void createProject(name.trim())}
                    className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-panel disabled:opacity-40"
                  >
                    Create
                  </button>
                </div>
                <div className="mt-3 flex items-center gap-4 text-xs text-zinc-400">
                  <button
                    onClick={() => void openFromDialog()}
                    disabled={loading}
                    className="underline-offset-2 hover:text-zinc-200 hover:underline"
                  >
                    …or open an existing project
                  </button>
                  <button
                    onClick={() => void openFromZip()}
                    disabled={loading}
                    className="underline-offset-2 hover:text-zinc-200 hover:underline"
                  >
                    import a .zip
                  </button>
                </div>
              </section>

              {error && (
                <p className="rounded-lg border border-red-900 bg-red-950/40 px-3 py-2 text-sm text-red-300">
                  {error}
                </p>
              )}
              {notice && (
                <p className="rounded-lg border border-green-900 bg-green-950/30 px-3 py-2 text-sm text-green-300">
                  {notice}
                </p>
              )}

              <section className="rounded-xl border border-border bg-surface p-5">
                <h2 className="mb-3 text-sm font-medium text-zinc-300">Recent</h2>
                {recents.length === 0 ? (
                  <p className="text-sm text-zinc-500">No recent projects yet.</p>
                ) : (
                  // Show ~10 projects (each row ≈ 36px) then scroll the rest.
                  <ul className="max-h-[360px] divide-y divide-border overflow-y-auto pr-1">
                    {recents.map((r) => (
                      <li key={r.path} className="flex items-center gap-2 py-1">
                        <button
                          onClick={() => void openByPath(r.path)}
                          className="flex min-w-0 flex-1 items-center justify-between gap-3 py-1 text-left hover:opacity-80"
                        >
                          <span className="shrink-0 text-sm text-zinc-200">{r.name}</span>
                          <span className="min-w-0 truncate text-xs text-zinc-500">{r.path}</span>
                        </button>
                        <button
                          onClick={() => void exportProject(r.path)}
                          disabled={exporting}
                          title="Export this project as a portable .zip"
                          className="shrink-0 rounded border border-border px-2 py-1 text-[11px] text-zinc-300 hover:bg-panel disabled:opacity-40"
                        >
                          {exportingPath === r.path ? 'Exporting…' : 'Export'}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>

            {/* Right column - getting started + demo */}
            <div className="flex flex-col gap-6">
              <GettingStartedCard />
              <FalKeyCard />
              <DemoSampleCard />
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto flex w-full max-w-4xl items-center justify-between pt-10 text-xs text-zinc-500">
        <button
          onClick={() => void studio().shell.openExternal('https://inlinestudio.art')}
          className="flex items-center gap-1.5 text-accent underline-offset-2 hover:underline"
        >
          inlinestudio.art
          <ExternalLinkIcon />
        </button>

        <div className="flex items-center gap-3">
          {currentVersion && <span>Inline Studio v{currentVersion}</span>}
          {updateAvailable && (
            <button
              onClick={() => void openReleases()}
              title="A newer version is available on GitHub"
              className="text-accent underline-offset-2 hover:underline"
            >
              Update available{updateVersion ? ` (v${updateVersion})` : ''} →
            </button>
          )}
          <button
            onClick={() => void studio().shell.openExternal('https://discord.gg/cSUS88VdY9')}
            title="Connect with the team on Discord"
            className="text-zinc-400 transition-colors hover:text-accent"
          >
            <DiscordIcon />
          </button>
        </div>
      </div>
    </div>
  )
}

function ExternalLinkIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="h-3.5 w-3.5"
    >
      <path d="M14 3h7v7" />
      <path d="M10 14 21 3" />
      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
    </svg>
  )
}

function DiscordIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
      <path d="M20.317 4.369a19.79 19.79 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.249a18.27 18.27 0 0 0-5.487 0 12.6 12.6 0 0 0-.617-1.25.077.077 0 0 0-.079-.036A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.1 13.1 0 0 1-1.872-.892.077.077 0 0 1-.008-.128c.126-.094.252-.192.372-.291a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.009c.12.099.246.198.373.292a.077.077 0 0 1-.006.127 12.3 12.3 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.84 19.84 0 0 0 6.002-3.03.077.077 0 0 0 .032-.056c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.028ZM8.02 15.331c-1.183 0-2.157-1.086-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.332-.956 2.418-2.157 2.418Zm7.975 0c-1.183 0-2.157-1.086-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.332-.946 2.418-2.157 2.418Z" />
    </svg>
  )
}
