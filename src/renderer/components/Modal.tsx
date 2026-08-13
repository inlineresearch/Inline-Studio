import { useEffect } from 'react'

/**
 * A small, non-blocking modal: a full-screen backdrop (mirrors the ContextMenu pattern)
 * plus a centered surface card. Dismisses on backdrop click and on Escape. Renders
 * nothing when closed.
 */
export function Modal({
  open,
  onClose,
  title,
  children,
  panelClassName = 'max-h-[85vh] w-full max-w-3xl',
  bodyClassName = 'min-h-0 flex-1 overflow-y-auto',
  headerAction,
}: {
  open: boolean
  onClose: () => void
  title?: string
  children: React.ReactNode
  /** Overrides the default sizing, for a dialog that wants the screen rather than its content. */
  panelClassName?: string
  /** Overrides the body's own scrolling, for a dialog that scrolls a section instead. */
  bodyClassName?: string
  /** Rendered in the header, before the close button: the dialog's primary action. */
  headerAction?: React.ReactNode
}): React.JSX.Element | null {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center bg-black/50 p-6"
      onClick={onClose}
    >
      <div
        className={`flex flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-2xl ${panelClassName}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold text-zinc-100">{title}</h2>
          <div className="flex items-center gap-3">
            {headerAction}
            <button
              onClick={onClose}
              title="Close"
              className="-m-1 rounded p-1 text-zinc-400 hover:text-zinc-100"
            >
              <CloseIcon />
            </button>
          </div>
        </div>
        <div className={bodyClassName}>{children}</div>
      </div>
    </div>
  )
}

function CloseIcon(): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}
