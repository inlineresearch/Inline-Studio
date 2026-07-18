import { useEffect, useState } from 'react'
import { useComfyStatusStore } from '../../store/comfyStatusStore'
import { useSettingsStore } from '../../store/settingsStore'
import { ComfyStatusPill } from '../../components/ComfyStatusPill'
import { ComfyUrlField } from '../../components/ComfyUrlField'
import { ComfyHelpDialog } from './ComfyHelpDialog'

/**
 * Home-screen ComfyUI connection bar. Non-blocking: it shows whether ComfyUI is reachable
 * and lets the user set + test the address inline, with friendly setup help in a popup.
 * The rest of the home screen stays usable whether or not ComfyUI is connected.
 */
export function ComfyConnectionBar(): React.JSX.Element {
  const running = useComfyStatusStore((s) => s.running)
  const checking = useComfyStatusStore((s) => s.checking)
  const start = useComfyStatusStore((s) => s.start)
  const test = useComfyStatusStore((s) => s.test)
  const loadSettings = useSettingsStore((s) => s.load)
  const connected = running === true
  const [helpOpen, setHelpOpen] = useState(false)
  const [tested, setTested] = useState<null | 'ok' | 'fail'>(null)

  // Load the saved URL and begin the reachability poll while the home screen is shown.
  useEffect(() => {
    void loadSettings()
  }, [loadSettings])
  useEffect(() => start(), [start])

  const onTest = async (): Promise<void> => {
    const ok = await test()
    setTested(ok ? 'ok' : 'fail')
  }

  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <div className="mb-3 flex items-center justify-between">
        <ComfyStatusPill />
        <button
          onClick={() => setHelpOpen(true)}
          className="text-xs text-accent underline-offset-2 hover:underline"
        >
          Need help connecting?
        </button>
      </div>

      <p className="mb-3 text-xs leading-relaxed text-zinc-400">
        Inline Studio renders through your own ComfyUI. Set its address, then test the connection.
      </p>

      <ComfyUrlField />
      <button
        onClick={() => void onTest()}
        disabled={checking}
        className="mt-2 w-full rounded border border-border px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-panel disabled:opacity-40"
      >
        {checking ? 'Testing…' : 'Test connection'}
      </button>

      {!connected && tested === 'fail' && (
        <p className="mt-2 text-xs text-zinc-400">
          Couldn’t reach ComfyUI at that address yet - open “Need help connecting?” for setup steps.
        </p>
      )}
      {!connected && tested === 'ok' && <p className="mt-2 text-xs text-green-400">Connected.</p>}

      <ComfyHelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </section>
  )
}
