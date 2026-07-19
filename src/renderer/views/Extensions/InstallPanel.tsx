/** Install from a repository URL, with live phase progress and failure detail. */
import { useState } from 'react'
import { useExtensionsStore } from '../../store/extensionsStore'
import { InstallSteps } from './InstallSteps'
import { SecurityReport } from './SecurityReport'

export function InstallPanel({ prefill }: { prefill?: string }): React.JSX.Element {
  const install = useExtensionsStore((s) => s.install)
  const beginInstall = useExtensionsStore((s) => s.beginInstall)
  const clearInstall = useExtensionsStore((s) => s.clearInstall)
  const canInstall = useExtensionsStore((s) => s.canInstall)
  const setTab = useExtensionsStore((s) => s.setTab)
  const tools = useExtensionsStore((s) => s.tools)
  const [source, setSource] = useState(prefill ?? '')
  const [ref, setRef] = useState('main')

  if (!canInstall) {
    const missing = tools.filter((t) => !t.available)
    return (
      <div className="flex flex-col gap-3 p-5">
        <h3 className="text-sm font-semibold text-zinc-100">Installing needs a couple of tools</h3>
        <p className="text-[11px] leading-relaxed text-zinc-500">
          Inline Studio uses these to download an extension and resolve its dependencies. Install
          them, then reopen this dialog.
        </p>
        {missing.map((tool) => (
          <div key={tool.name} className="rounded-md border border-border bg-surface/60 p-3">
            <div className="text-[12px] font-medium text-zinc-200">
              {tool.name} is not installed
            </div>
            <code className="mt-1 block whitespace-pre-wrap font-mono text-[11px] text-zinc-400">
              {tool.hint}
            </code>
          </div>
        ))}
      </div>
    )
  }

  // Consent gate takes over the panel: nothing was installed yet.
  if (install?.report && (install.report.needsConsent || install.report.blocked)) {
    return (
      <SecurityReport
        name={
          install.source
            .split('/')
            .pop()
            ?.replace(/\.git$/, '') ?? 'this extension'
        }
        report={install.report}
        onCancel={clearInstall}
        onApprove={(rules) => void beginInstall(install.source, install.ref, rules)}
      />
    )
  }

  if (install?.done) {
    const { done } = install
    return (
      <div className="flex flex-col gap-4 p-5">
        <div className="flex items-start gap-3 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3">
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/25 text-emerald-300">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              className="h-3 w-3"
            >
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </span>
          <div className="min-w-0">
            <div className="text-[13px] font-semibold text-emerald-200">
              Installed {done.name || done.extensionId}
            </div>
            <div className="mt-0.5 font-mono text-[11px] text-emerald-300/70">{done.version}</div>
            <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-400">
              {done.nodeTypes.length > 0
                ? `${done.nodeTypes.length} node${done.nodeTypes.length === 1 ? '' : 's'} added to the add-node menu: ${done.nodeTypes.join(', ')}`
                : 'No nodes are switched on yet. Enable them under Installed.'}
            </p>
            {done.restartRequired && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-amber-300">
                Restart Inline Studio to finish applying this version.
              </p>
            )}
          </div>
        </div>
        <InstallSteps phase="done" seen={install.seen} status="" />
        <div className="flex gap-2">
          <button
            onClick={() => setTab('installed')}
            className="rounded-md bg-emerald-500/90 px-3 py-1.5 text-[11px] font-semibold text-black hover:bg-emerald-400"
          >
            View installed
          </button>
          <button
            onClick={clearInstall}
            className="rounded-md border border-border px-3 py-1.5 text-[11px] font-medium text-zinc-300 hover:bg-surface hover:text-zinc-100"
          >
            Install another
          </button>
        </div>
      </div>
    )
  }

  const busy = install !== null && install.phase !== 'idle' && !install.error

  return (
    <div className="flex flex-col gap-3 p-5">
      <div className="flex flex-col gap-1">
        <label className="text-[11px] font-medium text-zinc-400">Repository URL</label>
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="https://github.com/author/inline-extension-demo"
          spellCheck={false}
          className="rounded-md border border-border bg-surface px-2.5 py-1.5 font-mono text-[12px] text-zinc-100 outline-none focus:border-zinc-500"
        />
      </div>
      <div className="flex items-end gap-2">
        <div className="flex w-40 flex-col gap-1">
          <label className="text-[11px] font-medium text-zinc-400">Tag or branch</label>
          <input
            value={ref}
            onChange={(e) => setRef(e.target.value)}
            spellCheck={false}
            className="rounded-md border border-border bg-surface px-2.5 py-1.5 font-mono text-[12px] text-zinc-100 outline-none focus:border-zinc-500"
          />
        </div>
        <button
          onClick={() => void beginInstall(source.trim(), ref.trim() || 'main')}
          disabled={!source.trim() || busy}
          className="rounded-md bg-emerald-500/90 px-3 py-1.5 text-[11px] font-semibold text-black hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? 'Installing…' : 'Install'}
        </button>
      </div>

      {busy && (
        <div className="rounded-md border border-border bg-surface/60 p-3">
          <InstallSteps phase={install.phase} seen={install.seen} status={install.status} />
        </div>
      )}

      {install?.error && install.phase !== 'idle' && (
        <div className="rounded-md border border-border bg-surface/60 p-3 opacity-80">
          <InstallSteps phase={install.phase} seen={install.seen} status="" failed />
        </div>
      )}

      {install?.error && (
        <div className="flex flex-col gap-2 rounded-md border border-red-500/40 bg-red-500/10 p-3">
          <div className="text-[12px] leading-relaxed text-red-300">{install.error}</div>
          {install.conflicts && install.conflicts.length > 0 && (
            <ul className="flex list-disc flex-col gap-1 pl-4">
              {install.conflicts.map((c) => (
                <li key={c.name} className="text-[11px] text-red-200/80">
                  {c.message}
                </li>
              ))}
            </ul>
          )}
          <button
            onClick={clearInstall}
            className="self-start rounded border border-red-500/40 px-2 py-1 text-[11px] text-red-200 hover:bg-red-500/10"
          >
            Dismiss
          </button>
        </div>
      )}

      <p className="text-[11px] leading-relaxed text-zinc-500">
        The repository is downloaded at the exact commit behind that tag and reviewed before
        anything runs. Its dependencies install into the extension's own folder, so they can never
        replace Inline's PyTorch or diffusers.
      </p>
    </div>
  )
}
