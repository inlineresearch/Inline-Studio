/** Registers all IPC handlers. New feature areas add their register* call here. */
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
import { registerUpdateHandlers } from './updates'
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
  registerUpdateHandlers()
  registerAppHandlers()
}
