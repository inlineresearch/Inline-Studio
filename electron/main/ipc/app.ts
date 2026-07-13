/** IPC for app-level metadata (running version, etc.). */
import { IpcChannels } from '@shared/ipc'
import { handle } from './handler'
import { caps } from '../capabilities'

export function registerAppHandlers(): void {
  handle<[], string>(IpcChannels.app.version, () => caps().appVersion())
}
