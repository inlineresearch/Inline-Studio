/**
 * Auto-update engine. All electron-updater knowledge lives here (engine isolation,
 * like comfy/ and export/): the rest of the app only sees `initAutoUpdater()` and the
 * two action wrappers, plus the `events:update*` broadcasts.
 *
 * Platform behaviour:
 * - Windows / Linux: download in the background, then emit `updateDownloaded` so the
 *   renderer can offer "Restart to install" (→ `quitAndInstall`).
 * - macOS: builds are unsigned, and Squirrel.Mac refuses to self-install an unsigned
 *   app. So we only detect (`autoDownload = false`) and emit `updateAvailable` with
 *   `notifyOnly: true`; the renderer opens the releases page instead.
 */
import { app } from 'electron'
import electronUpdater, { type ProgressInfo, type UpdateInfo } from 'electron-updater'
import { IpcChannels } from '@shared/ipc'
import { broadcast } from '../events/broadcaster'

// electron-updater is CJS; the autoUpdater singleton is on the default export.
// IMPORTANT: accessing `electronUpdater.autoUpdater` lazily constructs the platform
// updater (NsisUpdater on Windows), whose constructor immediately reads
// `app.getVersion()`. Touching it at module load — before `app` is ready — crashes
// with "Cannot read properties of undefined (reading 'getVersion')". So resolve it
// only inside the functions below, which all run after `app.whenReady()`.
function updater(): typeof electronUpdater.autoUpdater {
  return electronUpdater.autoUpdater
}

const isMac = process.platform === 'darwin'
/** Re-check this often while the app stays open. */
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000

/** Wire the updater once and kick off the first check. No-op (logged) outside a packaged app. */
export function initAutoUpdater(): void {
  if (!app.isPackaged) {
    console.log('[updater] skipping auto-update in dev (app not packaged)')
    return
  }

  const au = updater()
  // On macOS we can't self-install (unsigned), so never auto-download there.
  au.autoDownload = !isMac
  au.autoInstallOnAppQuit = !isMac

  au.on('update-available', (info: UpdateInfo) => {
    broadcast(IpcChannels.events.updateAvailable, { version: info.version, notifyOnly: isMac })
  })

  au.on('download-progress', (p: ProgressInfo) => {
    broadcast(IpcChannels.events.updateProgress, {
      percent: Math.round(p.percent),
      transferred: p.transferred,
      total: p.total,
    })
  })

  au.on('update-downloaded', (info: UpdateInfo) => {
    broadcast(IpcChannels.events.updateDownloaded, { version: info.version })
  })

  // A failed/interrupted update check must never crash the app.
  au.on('error', (err: Error) => {
    console.error('[updater]', err.message)
  })

  void checkForUpdates()
  setInterval(() => void checkForUpdates(), CHECK_INTERVAL_MS)
}

/** Check for a newer published release. Drives our own UI, so not `checkForUpdatesAndNotify`. */
export async function checkForUpdates(): Promise<void> {
  if (!app.isPackaged) return
  await updater().checkForUpdates()
}

/** Quit and install a downloaded update (Windows/Linux). */
export function quitAndInstall(): void {
  updater().quitAndInstall()
}
