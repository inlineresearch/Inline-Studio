import { useEffect, useState } from 'react'
import { useFalSettingsStore } from '../store/falSettingsStore'

/**
 * The fal.ai API key control for the closed-model generation engine, laid out for the Settings
 * panel. A secret, so it's write-only: once saved we only show "connected" (never echo the key
 * back). Saving stores it encrypted in main; "Clear" forgets it. Bring your own key from
 * https://fal.ai/dashboard/keys.
 */
export function FalKeyField(): React.JSX.Element {
  const configured = useFalSettingsStore((s) => s.configured)
  const encrypted = useFalSettingsStore((s) => s.encrypted)
  const error = useFalSettingsStore((s) => s.error)
  const load = useFalSettingsStore((s) => s.load)
  const setApiKey = useFalSettingsStore((s) => s.setApiKey)
  const clearApiKey = useFalSettingsStore((s) => s.clearApiKey)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    void load()
  }, [load])

  const save = async (): Promise<void> => {
    if (!draft.trim()) return
    const ok = await setApiKey(draft.trim())
    if (ok) setDraft('')
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-200">fal.ai API key</span>
        <span className="flex items-center gap-1.5">
          <span className={`h-2 w-2 rounded-full ${configured ? 'bg-green-500' : 'bg-zinc-600'}`} />
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">
            {configured ? 'Connected' : 'Not set'}
          </span>
        </span>
      </div>

      <p className="text-[11px] leading-relaxed text-zinc-500">
        Bring your own key to run the canvas generation nodes. Stored encrypted on this machine — it
        never leaves it. Get one from fal.ai/dashboard/keys.
      </p>

      <input
        type="password"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') void save()
        }}
        spellCheck={false}
        placeholder={configured ? '•••••••• saved' : 'Paste fal.ai API key'}
        className="w-full rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-zinc-200 outline-none placeholder:text-zinc-500 focus:border-accent"
      />

      <div className="flex items-center gap-2">
        <button
          onClick={() => void save()}
          disabled={!draft.trim()}
          className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-panel hover:brightness-110 disabled:opacity-40"
        >
          Save
        </button>
        {configured && (
          <button
            onClick={() => void clearApiKey()}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-zinc-400 hover:bg-surface"
          >
            Clear
          </button>
        )}
      </div>

      {error && <p className="text-[11px] text-red-400">{error}</p>}
      {configured && !encrypted && (
        <p className="text-[11px] text-amber-400">
          No OS keystore available — the key is stored as plaintext (0600).
        </p>
      )}
    </div>
  )
}
