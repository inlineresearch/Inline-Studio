/**
 * Registers the shell-agnostic IPC handlers shared by Electron and the headless web server.
 * Set the transport, broadcaster, and capabilities before calling this. New feature areas add
 * their register* call here. Electron-only handlers (auto-update) are registered by the Electron
 * entry directly, so this list stays runnable on the server.
 */
import { registerProjectHandlers } from './project'
import { registerAssetHandlers } from './assets'
import { registerFolderHandlers } from './folders'
import { registerMoodboardHandlers } from './moodboard'
import { registerTimelineHandlers } from './timeline'
import { registerFrameHandlers } from './frames'
import { registerComfyHandlers } from './comfy'
import { registerSettingsHandlers } from './settings'
import { registerCoreHandlers } from './core'
import { registerFalSettingsHandlers } from './falSettings'
import { registerGenerationHandlers } from './generation'
import { registerExportHandlers } from './export'
import { registerClipboardHandlers } from './clipboard'
import { registerMediaHandlers } from './media'
import { registerShellHandlers } from './shell'
import { registerAppHandlers } from './app'

export function registerIpcHandlers(): void {
  registerProjectHandlers()
  registerAssetHandlers()
  registerFolderHandlers()
  registerMoodboardHandlers()
  registerTimelineHandlers()
  registerFrameHandlers()
  registerComfyHandlers()
  registerSettingsHandlers()
  registerCoreHandlers()
  registerFalSettingsHandlers()
  registerGenerationHandlers()
  registerExportHandlers()
  registerClipboardHandlers()
  registerMediaHandlers()
  registerShellHandlers()
  registerAppHandlers()
}
