import { useEffect, useState } from 'react'
import { InfoTip } from './InfoTip'
import { useHfSettingsStore } from '../store/hfSettingsStore'

/** The Hugging Face token, write-only like the fal key; gated repos such as Klein 9B need it. */
export function HfTokenField(): React.JSX.Element {
  const configured = useHfSettingsStore((s) => s.configured)
  const error = useHfSettingsStore((s) => s.error)
  const load = useHfSettingsStore((s) => s.load)
  const setToken = useHfSettingsStore((s) => s.setToken)
  const clearToken = useHfSettingsStore((s) => s.clearToken)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    void load()
  }, [load])

  const save = async (): Promise<void> => {
    if (!draft.trim()) return
    const ok = await setToken(draft.trim())
    if (ok) setDraft('')
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-zinc-200">Hugging Face token</span>
          <InfoTip label="the Hugging Face token">
            Only needed for models behind a licence, such as FLUX.2 Klein 9B. Accept the licence on
            the model&rsquo;s page first, then paste a token from huggingface.co/settings/tokens -
            read access is enough. Stored on the machine running the engine, in a file only your
            user account can read, and never sent back to the browser.
          </InfoTip>
        </span>
        <span className="flex items-center gap-1.5">
          <span className={`h-2 w-2 rounded-full ${configured ? 'bg-green-500' : 'bg-zinc-600'}`} />
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">
            {configured ? 'Connected' : 'Not set'}
          </span>
        </span>
      </div>

      <input
        type="password"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') void save()
        }}
        spellCheck={false}
        placeholder={configured ? '•••••••• saved' : 'Paste Hugging Face token'}
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
            onClick={() => void clearToken()}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-zinc-400 hover:bg-surface"
          >
            Clear
          </button>
        )}
      </div>

      {error && <p className="text-[11px] text-red-400">{error}</p>}
    </div>
  )
}
