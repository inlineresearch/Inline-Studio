import { FalKeyField } from '../../components/FalKeyField'
import { CloseIcon, SettingsIcon } from '../../components/icons'
import { useCanvasPrefsStore, type EdgeStyle } from '../../store/canvasPrefsStore'

/**
 * Right-hand Settings sidebar, opened from the workspace header's gear button. Holds
 * app/account configuration - the fal.ai API key, plus per-device canvas display preferences.
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

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <section className="rounded-lg border border-border bg-panel/40 p-3">
          <FalKeyField />
        </section>
        <section className="rounded-lg border border-border bg-panel/40 p-3">
          <ConnectionStyleField />
        </section>
      </div>
    </div>
  )
}

/** Picks how connectors between nodes are drawn; applies to every edge live. */
function ConnectionStyleField(): React.JSX.Element {
  const edgeStyle = useCanvasPrefsStore((s) => s.edgeStyle)
  const setEdgeStyle = useCanvasPrefsStore((s) => s.setEdgeStyle)
  const options: { value: EdgeStyle; label: string; hint: string; preview: React.JSX.Element }[] = [
    { value: 'angled', label: 'Angled', hint: 'Cornered', preview: <EdgePreview kind="angled" /> },
    { value: 'wave', label: 'Wave', hint: 'Curved', preview: <EdgePreview kind="wave" /> },
  ]
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col">
        <span className="text-xs font-medium text-zinc-200">Connection line</span>
        <span className="text-[11px] text-zinc-500">Line style for links between nodes.</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {options.map((o) => {
          const active = edgeStyle === o.value
          return (
            <button
              key={o.value}
              onClick={() => setEdgeStyle(o.value)}
              aria-pressed={active}
              className={`flex flex-col items-center gap-1.5 rounded-md border px-2 py-2 transition-colors ${
                active
                  ? 'border-accent bg-accent/10 text-zinc-100'
                  : 'border-border text-zinc-400 hover:border-zinc-500 hover:text-zinc-200'
              }`}
            >
              <span className={active ? 'text-accent' : 'text-zinc-500'}>{o.preview}</span>
              <span className="text-[11px] font-medium">{o.label}</span>
              <span className="text-[10px] text-zinc-500">{o.hint}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/** Tiny SVG showing the two node-connection line styles. */
function EdgePreview({ kind }: { kind: EdgeStyle }): React.JSX.Element {
  return (
    <svg viewBox="0 0 48 20" fill="none" className="h-5 w-12" stroke="currentColor" strokeWidth="2">
      {kind === 'angled' ? (
        <path d="M2 4 H24 V16 H46" strokeLinejoin="round" strokeLinecap="round" />
      ) : (
        <path d="M2 4 C20 4 28 16 46 16" strokeLinecap="round" />
      )}
    </svg>
  )
}
