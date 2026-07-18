/**
 * App-wide auto-update banner. Pinned top-center; hidden until the main process
 * reports an update. Windows/Linux download in the background and offer a restart;
 * macOS (unsigned) links out to the releases page.
 */
import { useUpdateStore } from '../store/updateStore'

export function UpdateBanner(): React.JSX.Element | null {
  const status = useUpdateStore((s) => s.status)
  const version = useUpdateStore((s) => s.version)
  const percent = useUpdateStore((s) => s.percent)
  const install = useUpdateStore((s) => s.install)
  const openReleases = useUpdateStore((s) => s.openReleases)

  if (status === 'idle') return null

  const v = version ? `Inline Studio ${version}` : 'A new version'

  return (
    <div className="pointer-events-none fixed left-0 right-0 top-3 z-50 flex justify-center">
      <div className="pointer-events-auto flex items-center gap-3 rounded-lg border border-accent/40 bg-surface px-4 py-2 text-sm text-zinc-200 shadow-lg">
        {status === 'downloading' ? (
          <>
            <span>Downloading update… {percent}%</span>
            <span className="h-1 w-24 overflow-hidden rounded bg-panel">
              <span
                className="block h-full bg-accent transition-all"
                style={{ width: `${percent}%` }}
              />
            </span>
          </>
        ) : status === 'ready' ? (
          <>
            <span>{v} is ready.</span>
            <button
              onClick={() => void install()}
              className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-panel hover:opacity-90"
            >
              Restart to install
            </button>
          </>
        ) : (
          // 'available' - macOS notify-only path.
          <>
            <span>{v} is available.</span>
            <button
              onClick={() => void openReleases()}
              className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-panel hover:opacity-90"
            >
              Download
            </button>
          </>
        )}
      </div>
    </div>
  )
}
