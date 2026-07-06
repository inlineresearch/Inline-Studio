import { FalKeyField } from '../../components/FalKeyField'
import { CloseIcon, SettingsIcon } from '../../components/icons'

/**
 * Right-hand Settings sidebar, opened from the workspace header's gear button. Holds
 * app/account configuration — starting with the fal.ai API key for the generation engine.
 */
export function SettingsPanel({ onClose }: { onClose: () => void }): React.JSX.Element {
  return (
    <div className="flex h-full flex-col border-l border-border bg-surface">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-border px-3">
        <div className="flex items-center gap-2 text-zinc-200">
          <SettingsIcon className="h-4 w-4" />
          <span className="text-sm font-semibold">Settings</span>
        </div>
        <button
          onClick={onClose}
          title="Close settings"
          aria-label="Close settings"
          className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 hover:bg-panel hover:text-zinc-100"
        >
          <CloseIcon className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <section className="rounded-lg border border-border bg-panel/40 p-3">
          <FalKeyField />
        </section>
      </div>
    </div>
  )
}
