import { CheckIcon } from '../../components/icons'
import { XIcon } from './nodes/NodeBadge'

/**
 * Shared header for the node settings sidebars (fal Generate / Inline Core). Shows the node title
 * and, when there are unsaved edits, an emerald **Update** button (also bound to ⌘/Ctrl+S) that
 * persists them. Params draft locally and only commit on Update / close, so nothing is lost.
 */
export function SettingsHeader({
  title,
  dirty,
  onUpdate,
  onClose,
}: {
  title: string
  dirty: boolean
  onUpdate: () => void
  onClose: () => void
}): React.JSX.Element {
  return (
    <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
      <div className="flex min-w-0 flex-col">
        <span className="text-xs font-semibold text-zinc-100">Settings</span>
        <span className="truncate text-[10px] text-zinc-500">{title}</span>
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        {dirty && (
          <button
            onClick={onUpdate}
            title="Apply changes (⌘/Ctrl+S)"
            className="flex h-6 items-center gap-1 rounded-md bg-emerald-500/90 px-2 text-[11px] font-medium text-white shadow-sm hover:bg-emerald-500"
          >
            <CheckIcon className="h-3 w-3" />
            Update
          </button>
        )}
        <button
          onClick={onClose}
          className="flex h-6 w-6 items-center justify-center rounded text-zinc-400 hover:bg-surface hover:text-zinc-100"
          aria-label="Close settings"
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </div>
    </header>
  )
}
