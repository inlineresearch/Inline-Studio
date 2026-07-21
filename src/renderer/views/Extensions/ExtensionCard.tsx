/**
 * One installed extension. Collapsed it shows identity and provenance - name, ref, description,
 * license, repository, and whether the ref has moved upstream. Nodes live behind the disclosure so
 * a list of many extensions stays scannable.
 */
import { useState } from 'react'
import type { ExtensionInfo } from '@shared/extensions'
import { useExtensionsStore } from '../../store/extensionsStore'

function Toggle({
  on,
  onChange,
  label,
}: {
  on: boolean
  onChange: (next: boolean) => void
  label: string
}): React.JSX.Element {
  return (
    <button
      role="switch"
      aria-checked={on}
      aria-label={label}
      onClick={() => onChange(!on)}
      className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${
        on ? 'bg-emerald-500/80' : 'bg-zinc-700'
      }`}
    >
      <span
        className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${
          on ? 'left-3.5' : 'left-0.5'
        }`}
      />
    </button>
  )
}

function ChevronIcon({ open }: { open: boolean }): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-90' : ''}`}
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  )
}

function ExternalIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3 w-3"
    >
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  )
}

function TrashIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-3.5 w-3.5"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  )
}

/** Read the link as a repository rather than a raw URL. */
function repoLabel(repo: string): string {
  return repo
    .replace(/^https:\/\//, '')
    .replace(/^file:\/\/\//, '')
    .replace(/^git@/, '')
    .replace(/\.git$/, '')
}

export function ExtensionCard({ extension }: { extension: ExtensionInfo }): React.JSX.Element {
  const setEnabled = useExtensionsStore((s) => s.setEnabled)
  const setNodeEnabled = useExtensionsStore((s) => s.setNodeEnabled)
  const switchVersion = useExtensionsStore((s) => s.switchVersion)
  const uninstall = useExtensionsStore((s) => s.uninstall)
  const update = useExtensionsStore((s) => s.updates[extension.extensionId])
  const [expanded, setExpanded] = useState(false)
  const [confirming, setConfirming] = useState(false)

  const others = extension.versions.filter((v) => v !== extension.installed)
  const enabledNodes = extension.nodes.filter((n) => n.enabled).length

  return (
    <div className="flex flex-col rounded-lg border border-border bg-panel/40">
      <div className="flex items-start gap-3 p-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="truncate text-[13px] font-semibold text-zinc-100">
              {extension.name}
            </span>
            {extension.ref && (
              <span className="shrink-0 rounded bg-surface px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                {extension.ref}
              </span>
            )}
            {update?.behind && (
              <span
                title={
                  update.latestTag && update.latestTag !== extension.ref
                    ? `${update.latestTag} has been published`
                    : `Installed ${update.installedSha}, upstream is at ${update.remoteSha}`
                }
                className="shrink-0 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-300"
              >
                {update.latestTag && update.latestTag !== extension.ref
                  ? `${update.latestTag} available`
                  : 'Update available'}
              </span>
            )}
          </div>

          {extension.description && (
            <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-zinc-500">
              {extension.description}
            </p>
          )}

          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-zinc-500">
            {extension.license && <span>{extension.license}</span>}
            {extension.sha && (
              <span className="font-mono" title="Installed commit">
                {extension.sha}
                {update?.behind && update.remoteSha && (
                  <span className="text-amber-400/80"> → {update.remoteSha}</span>
                )}
              </span>
            )}
            {extension.repo && (
              <a
                href={extension.repo}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1 truncate text-zinc-500 hover:text-zinc-300"
              >
                {repoLabel(extension.repo)}
                <ExternalIcon />
              </a>
            )}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Toggle
            on={extension.enabled}
            onChange={(next) => void setEnabled(extension.extensionId, next)}
            label={`Enable ${extension.name}`}
          />
          {confirming ? (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-zinc-400">Remove?</span>
              <button
                onClick={() => setConfirming(false)}
                className="rounded border border-border px-2 py-1 text-[11px] text-zinc-300 hover:bg-surface"
              >
                No
              </button>
              <button
                onClick={() => {
                  setConfirming(false)
                  void uninstall(extension.extensionId)
                }}
                className="rounded bg-red-500/80 px-2 py-1 text-[11px] font-semibold text-black hover:bg-red-400"
              >
                Uninstall
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              aria-label={`Uninstall ${extension.name}`}
              title="Uninstall"
              className="rounded-md border border-border p-1.5 text-zinc-400 hover:border-red-500/40 hover:text-red-300"
            >
              <TrashIcon />
            </button>
          )}
        </div>
      </div>

      <button
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex items-center gap-1.5 border-t border-border px-3 py-2 text-left text-[11px] text-zinc-400 hover:text-zinc-200"
      >
        <ChevronIcon open={expanded} />
        {extension.nodes.length} node{extension.nodes.length === 1 ? '' : 's'}
        <span className="text-zinc-600">({enabledNodes} on)</span>
      </button>

      {expanded && (
        <div className="flex flex-col gap-1.5 border-t border-border px-3 py-2.5">
          {extension.nodes.map((node) => (
            <div key={node.type} className="flex items-center gap-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-[12px] text-zinc-200">{node.title}</div>
                <div className="truncate font-mono text-[10px] text-zinc-500">{node.type}</div>
              </div>
              <Toggle
                on={node.enabled && extension.enabled}
                onChange={(next) => void setNodeEnabled(extension.extensionId, node.type, next)}
                label={`Enable ${node.title}`}
              />
            </div>
          ))}

          {others.length > 0 ? (
            <div className="mt-1 flex items-center gap-2 border-t border-border pt-2.5">
              <select
                value=""
                onChange={(e) => {
                  if (e.target.value) void switchVersion(extension.extensionId, e.target.value)
                }}
                className="rounded-md border border-border bg-surface px-2 py-1 text-[11px] text-zinc-300 outline-none"
              >
                <option value="">Roll back…</option>
                {others.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="mt-1 border-t border-border pt-2.5">
              <span className="font-mono text-[10px] text-zinc-600">{extension.installed}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
