/** IPC handlers for the Inline Core engine: status + node descriptors (the palette). */
import { IpcChannels } from '@shared/ipc'
import type { CoreModels } from '@shared/coreNodes'
import type { CoreStatus } from '@shared/types'
import { handle } from './handler'
import { pingCore, fetchModels } from '../core/client'

export function registerCoreHandlers(): void {
  handle<[], CoreStatus>(IpcChannels.core.status, () => pingCore())
  handle<[], CoreModels>(IpcChannels.core.models, () => fetchModels())
}
